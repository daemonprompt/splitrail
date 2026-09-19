use std::collections::{BTreeMap, HashMap, HashSet};

use rmcp::handler::server::router::tool::ToolRouter;
use rmcp::handler::server::wrapper::Parameters;
use rmcp::model::{
    AnnotateAble, Implementation, ListResourcesResult, PaginatedRequestParam, ProtocolVersion,
    RawResource, ReadResourceRequestParam, ReadResourceResult, ResourceContents,
    ServerCapabilities, ServerInfo,
};
use rmcp::service::RequestContext;
use rmcp::{
    ErrorData as McpError, Json, RoleServer, ServerHandler, ServiceExt, tool, tool_handler,
    tool_router,
};

use crate::analyzers::claude_code::claude_projects_dir;
use crate::types::{ConversationMessage, MultiAnalyzerStats};
use crate::{create_analyzer_registry, utils};

use super::types::*;

/// Resource URI constants
mod resource_uris {
    pub const DAILY_SUMMARY: &str = "splitrail://summary";
    pub const MODEL_BREAKDOWN: &str = "splitrail://models";
}

/// The Splitrail MCP Server
#[derive(Clone)]
pub struct SplitrailMcpServer {
    tool_router: ToolRouter<Self>,
}

impl SplitrailMcpServer {
    pub fn new() -> Self {
        Self {
            tool_router: Self::tool_router(),
        }
    }

    /// Load stats from all analyzers (reuses existing infrastructure)
    fn load_stats(&self) -> Result<MultiAnalyzerStats, McpError> {
        let registry = create_analyzer_registry();
        registry
            .load_all_stats_parallel()
            .map_err(|e| McpError::internal_error(format!("Failed to load stats: {}", e), None))
    }

    /// Compute file operation stats grouped by date from raw messages.
    /// This is computed on-demand since DailyStats only contains TUI-relevant fields.
    fn compute_file_ops_by_date(
        stats: &MultiAnalyzerStats,
        analyzer: Option<&str>,
    ) -> HashMap<String, DateFileOps> {
        let messages: Vec<_> = if let Some(analyzer_name) = analyzer {
            stats
                .analyzer_stats
                .iter()
                .filter(|a| a.analyzer_name.eq_ignore_ascii_case(analyzer_name))
                .flat_map(|a| a.messages.iter())
                .collect()
        } else {
            stats
                .analyzer_stats
                .iter()
                .flat_map(|a| a.messages.iter())
                .collect()
        };

        let mut file_ops_by_date: HashMap<String, DateFileOps> = HashMap::new();
        for msg in messages {
            let date = msg
                .date
                .with_timezone(&chrono::Local)
                .format("%Y-%m-%d")
                .to_string();
            let entry = file_ops_by_date.entry(date).or_default();
            entry.files_read += msg.stats.files_read;
            entry.files_edited += msg.stats.files_edited;
            entry.files_added += msg.stats.files_added;
            entry.terminal_commands += msg.stats.terminal_commands;
        }
        file_ops_by_date
    }

    /// Resolve the set of conversation_hashes that correspond to a given session UUID.
    ///
    /// Claude Code stores sessions at:
    ///   ~/.claude/projects/{PROJECT_ID}/{SESSION_UUID}.jsonl
    ///   ~/.claude/projects/{PROJECT_ID}/{SESSION_UUID}/subagents/**/*.jsonl
    ///
    /// The conversation_hash stored on each ConversationMessage is derived by hashing
    /// the full path with crate::utils::hash_text. We scan the projects directory for
    /// files whose stem contains the session_id, then hash their paths to get the
    /// conversation_hash keys we can filter on.
    fn conversation_hashes_for_session(session_id: &str) -> HashSet<String> {
        let mut hashes = HashSet::new();
        let config_dir = std::env::var_os("CLAUDE_CONFIG_DIR");
        let home_dir = dirs::home_dir();
        let projects_dir = match claude_projects_dir(config_dir.as_deref(), home_dir.as_deref()) {
            Some(d) if d.is_dir() => d,
            _ => return hashes,
        };

        use walkdir::WalkDir;
        for entry in WalkDir::new(&projects_dir)
            .into_iter()
            .filter_map(|e| e.ok())
        {
            let path = entry.path();
            if path.is_file() {
                if let Some(stem) = path.file_stem().and_then(|s| s.to_str()) {
                    if stem.contains(session_id) {
                        let hash = utils::hash_text(&path.to_string_lossy());
                        hashes.insert(hash);
                    }
                }
            }
        }
        hashes
    }

    /// Filter messages to those belonging to sessions matching session_id or bead_id.
    ///
    /// - session_id: matches JSONL files whose stem contains the UUID string
    /// - bead_id: matches messages where session_name contains the bead_id string
    /// - If session_id is provided it takes precedence over bead_id.
    /// - If neither is provided, all messages are returned unchanged.
    fn filter_messages_by_session<'a>(
        messages: &'a [ConversationMessage],
        session_id: Option<&str>,
        bead_id: Option<&str>,
    ) -> Vec<&'a ConversationMessage> {
        if let Some(sid) = session_id {
            let hashes = Self::conversation_hashes_for_session(sid);
            messages
                .iter()
                .filter(|m| hashes.contains(&m.conversation_hash))
                .collect()
        } else if let Some(bid) = bead_id {
            messages
                .iter()
                .filter(|m| {
                    m.session_name
                        .as_deref()
                        .is_some_and(|name| name.contains(bid))
                })
                .collect()
        } else {
            messages.iter().collect()
        }
    }

    /// Get daily stats for a specific analyzer or combined across all
    fn get_daily_stats_for_analyzer(
        stats: &MultiAnalyzerStats,
        analyzer: Option<&str>,
    ) -> BTreeMap<String, crate::types::DailyStats> {
        if let Some(analyzer_name) = analyzer {
            // Find specific analyzer
            for analyzer_stats in &stats.analyzer_stats {
                if analyzer_stats
                    .analyzer_name
                    .eq_ignore_ascii_case(analyzer_name)
                {
                    return analyzer_stats.daily_stats.clone();
                }
            }
            BTreeMap::new()
        } else {
            // Combine all messages and aggregate
            let all_messages: Vec<_> = stats
                .analyzer_stats
                .iter()
                .flat_map(|s| s.messages.iter().cloned())
                .collect();
            utils::aggregate_by_date(&all_messages)
        }
    }
}

#[tool_router]
impl SplitrailMcpServer {
    #[tool(
        name = "get_daily_stats",
        description = "Get daily usage statistics including messages, costs, tokens, and file operations. Can filter by date, analyzer, limit to recent days, or scope to a specific session_id or bead_id."
    )]
    async fn get_daily_stats(
        &self,
        Parameters(req): Parameters<GetDailyStatsRequest>,
    ) -> Result<Json<DailyStatsResponse>, String> {
        let stats = self.load_stats().map_err(|e| e.to_string())?;

        // When session_id or bead_id is provided, filter raw messages and re-aggregate.
        // The date and limit filters are not applied in this path.
        if req.session_id.is_some() || req.bead_id.is_some() {
            let all_messages: Vec<ConversationMessage> = stats
                .analyzer_stats
                .iter()
                .filter(|a| {
                    req.analyzer
                        .as_deref()
                        .map(|name| a.analyzer_name.eq_ignore_ascii_case(name))
                        .unwrap_or(true)
                })
                .flat_map(|a| a.messages.iter().cloned())
                .collect();

            let filtered_refs = Self::filter_messages_by_session(
                &all_messages,
                req.session_id.as_deref(),
                req.bead_id.as_deref(),
            );
            let filtered: Vec<ConversationMessage> = filtered_refs.into_iter().cloned().collect();
            let daily_stats = utils::aggregate_by_date(&filtered);
            let file_ops_by_date = {
                let mut map: HashMap<String, DateFileOps> = HashMap::new();
                for msg in &filtered {
                    let date = msg
                        .date
                        .with_timezone(&chrono::Local)
                        .format("%Y-%m-%d")
                        .to_string();
                    let entry = map.entry(date).or_default();
                    entry.files_read += msg.stats.files_read;
                    entry.files_edited += msg.stats.files_edited;
                    entry.files_added += msg.stats.files_added;
                    entry.terminal_commands += msg.stats.terminal_commands;
                }
                map
            };
            let mut results: Vec<DailySummary> = daily_stats
                .iter()
                .map(|(date, ds)| {
                    let file_ops = file_ops_by_date.get(date).cloned().unwrap_or_default();
                    DailySummary::new(date.as_str(), ds, &file_ops)
                })
                .collect();
            results.sort_by(|a, b| b.date.cmp(&a.date));
            return Ok(Json(DailyStatsResponse { results }));
        }

        let daily_stats = Self::get_daily_stats_for_analyzer(&stats, req.analyzer.as_deref());
        let file_ops_by_date = Self::compute_file_ops_by_date(&stats, req.analyzer.as_deref());

        let mut results: Vec<DailySummary> = if let Some(ref date) = req.date {
            // Filter by specific date
            daily_stats.get(date).map_or_else(Vec::new, |ds| {
                let file_ops = file_ops_by_date.get(date).cloned().unwrap_or_default();
                vec![DailySummary::new(date.as_str(), ds, &file_ops)]
            })
        } else {
            // All dates
            daily_stats
                .iter()
                .map(|(date, ds)| {
                    let file_ops = file_ops_by_date.get(date).cloned().unwrap_or_default();
                    DailySummary::new(date.as_str(), ds, &file_ops)
                })
                .collect()
        };

        // Sort by date descending (most recent first)
        results.sort_by(|a, b| b.date.cmp(&a.date));

        // Apply limit if specified
        if let Some(limit) = req.limit {
            results.truncate(limit);
        }

        Ok(Json(DailyStatsResponse { results }))
    }

    #[tool(
        name = "get_model_usage",
        description = "Get breakdown of AI model usage across conversations. Shows which models were used and how many messages each generated."
    )]
    async fn get_model_usage(
        &self,
        Parameters(req): Parameters<GetModelUsageRequest>,
    ) -> Result<Json<ModelUsageResponse>, String> {
        let stats = self.load_stats().map_err(|e| e.to_string())?;
        let daily_stats = Self::get_daily_stats_for_analyzer(&stats, req.analyzer.as_deref());

        let mut model_counts: HashMap<String, u32> = HashMap::new();

        if let Some(date) = req.date {
            if let Some(ds) = daily_stats.get(&date) {
                for (model, count) in &ds.models {
                    *model_counts.entry(model.clone()).or_insert(0) += count;
                }
            }
        } else {
            for ds in daily_stats.values() {
                for (model, count) in &ds.models {
                    *model_counts.entry(model.clone()).or_insert(0) += count;
                }
            }
        }

        let total_messages: u32 = model_counts.values().sum();
        let mut models: Vec<ModelUsageEntry> = model_counts
            .into_iter()
            .map(|(model, count)| ModelUsageEntry {
                model,
                message_count: count,
            })
            .collect();
        models.sort_by_key(|b| std::cmp::Reverse(b.message_count));

        Ok(Json(ModelUsageResponse {
            models,
            total_messages,
        }))
    }

    #[tool(
        name = "get_cost_breakdown",
        description = "Get cost breakdown over a date range. Shows total cost, daily costs, and average daily spending."
    )]
    async fn get_cost_breakdown(
        &self,
        Parameters(req): Parameters<GetCostBreakdownRequest>,
    ) -> Result<Json<CostBreakdownResponse>, String> {
        let stats = self.load_stats().map_err(|e| e.to_string())?;
        let daily_stats = Self::get_daily_stats_for_analyzer(&stats, req.analyzer.as_deref());

        let daily_costs: Vec<DailyCost> = daily_stats
            .iter()
            .filter(|(date, _)| {
                let after_start = req
                    .start_date
                    .as_ref()
                    .map(|s| date.as_str() >= s.as_str())
                    .unwrap_or(true);
                let before_end = req
                    .end_date
                    .as_ref()
                    .map(|e| date.as_str() <= e.as_str())
                    .unwrap_or(true);
                after_start && before_end
            })
            .map(|(date, ds)| DailyCost {
                date: date.clone(),
                cost: ds.stats.cost(),
            })
            .collect();

        let total_cost: f64 = daily_costs.iter().map(|dc| dc.cost).sum();
        let average_daily_cost = if daily_costs.is_empty() {
            0.0
        } else {
            total_cost / daily_costs.len() as f64
        };

        Ok(Json(CostBreakdownResponse {
            total_cost,
            daily_costs,
            average_daily_cost,
        }))
    }

    #[tool(
        name = "get_file_operations",
        description = "Get file operation statistics including reads, writes, edits, and terminal commands. Can filter by date, analyzer, session_id, or bead_id."
    )]
    async fn get_file_operations(
        &self,
        Parameters(req): Parameters<GetFileOpsRequest>,
    ) -> Result<Json<FileOpsResponse>, String> {
        let stats = self.load_stats().map_err(|e| e.to_string())?;

        // Collect messages, optionally filtered by analyzer
        let all_messages: Vec<ConversationMessage> = if let Some(ref analyzer_name) = req.analyzer {
            stats
                .analyzer_stats
                .iter()
                .filter(|a| a.analyzer_name.eq_ignore_ascii_case(analyzer_name))
                .flat_map(|a| a.messages.iter().cloned())
                .collect()
        } else {
            stats
                .analyzer_stats
                .iter()
                .flat_map(|a| a.messages.iter().cloned())
                .collect()
        };

        // Apply session_id or bead_id filter first, then date filter
        let session_filtered: Vec<ConversationMessage> = if req.session_id.is_some()
            || req.bead_id.is_some()
        {
            Self::filter_messages_by_session(
                &all_messages,
                req.session_id.as_deref(),
                req.bead_id.as_deref(),
            )
            .into_iter()
            .cloned()
            .collect()
        } else {
            all_messages
        };

        // Filter by date if specified
        let filtered: Vec<&ConversationMessage> = if let Some(ref date) = req.date {
            session_filtered
                .iter()
                .filter(|m| {
                    m.date
                        .with_timezone(&chrono::Local)
                        .format("%Y-%m-%d")
                        .to_string()
                        == *date
                })
                .collect()
        } else {
            session_filtered.iter().collect()
        };

        // Sum file operations from raw Stats
        let mut response = FileOpsResponse::default();
        for msg in filtered {
            response.files_read += msg.stats.files_read;
            response.files_edited += msg.stats.files_edited;
            response.files_added += msg.stats.files_added;
            response.files_deleted += msg.stats.files_deleted;
            response.lines_read += msg.stats.lines_read;
            response.lines_edited += msg.stats.lines_edited;
            response.lines_added += msg.stats.lines_added;
            response.lines_deleted += msg.stats.lines_deleted;
            response.bytes_read += msg.stats.bytes_read;
            response.bytes_edited += msg.stats.bytes_edited;
            response.bytes_added += msg.stats.bytes_added;
            response.bytes_deleted += msg.stats.bytes_deleted;
            response.terminal_commands += msg.stats.terminal_commands;
            response.file_searches += msg.stats.file_searches;
            response.file_content_searches += msg.stats.file_content_searches;
        }

        Ok(Json(response))
    }

    #[tool(
        name = "compare_tools",
        description = "Compare usage across different AI coding tools (Claude Code, Codex CLI, Gemini CLI, etc). Shows cost, messages, and activity for each."
    )]
    async fn compare_tools(
        &self,
        Parameters(req): Parameters<CompareToolsRequest>,
    ) -> Result<Json<ToolComparisonResponse>, String> {
        let stats = self.load_stats().map_err(|e| e.to_string())?;

        let tools: Vec<ToolSummary> = stats
            .analyzer_stats
            .iter()
            .map(|analyzer_stats| {
                let filtered_stats: Vec<_> = analyzer_stats
                    .daily_stats
                    .iter()
                    .filter(|(date, _)| {
                        let after_start = req
                            .start_date
                            .as_ref()
                            .map(|s| date.as_str() >= s.as_str())
                            .unwrap_or(true);
                        let before_end = req
                            .end_date
                            .as_ref()
                            .map(|e| date.as_str() <= e.as_str())
                            .unwrap_or(true);
                        after_start && before_end
                    })
                    .collect();

                let total_cost: f64 = filtered_stats.iter().map(|(_, ds)| ds.stats.cost()).sum();
                let total_messages: u64 = filtered_stats
                    .iter()
                    .map(|(_, ds)| (ds.user_messages + ds.ai_messages) as u64)
                    .sum();
                let total_conversations: u64 = filtered_stats
                    .iter()
                    .map(|(_, ds)| ds.conversations as u64)
                    .sum();
                let total_tokens: u64 = filtered_stats
                    .iter()
                    .map(|(_, ds)| ds.stats.input_tokens + ds.stats.output_tokens)
                    .sum();
                let total_tool_calls: u32 = filtered_stats
                    .iter()
                    .map(|(_, ds)| ds.stats.tool_calls)
                    .sum();

                ToolSummary {
                    name: analyzer_stats.analyzer_name.clone(),
                    total_cost,
                    total_messages,
                    total_conversations,
                    total_tokens,
                    total_tool_calls,
                }
            })
            .collect();

        Ok(Json(ToolComparisonResponse { tools }))
    }

    #[tool(
        name = "list_analyzers",
        description = "List all available AI coding tool analyzers (e.g., Claude Code, Codex CLI, Gemini CLI, GitHub Copilot, GitHub Copilot CLI)."
    )]
    async fn list_analyzers(
        &self,
        Parameters(_req): Parameters<ListAnalyzersRequest>,
    ) -> Result<Json<AnalyzerListResponse>, String> {
        let registry = create_analyzer_registry();
        let analyzers: Vec<String> = registry
            .available_analyzers()
            .into_iter()
            .map(|a| a.display_name().to_string())
            .collect();

        Ok(Json(AnalyzerListResponse { analyzers }))
    }
}

#[tool_handler]
impl ServerHandler for SplitrailMcpServer {
    fn get_info(&self) -> ServerInfo {
        ServerInfo {
            protocol_version: ProtocolVersion::LATEST,
            capabilities: ServerCapabilities::builder()
                .enable_tools()
                .enable_resources()
                .build(),
            server_info: Implementation {
                name: "splitrail".to_string(),
                version: env!("CARGO_PKG_VERSION").to_string(),
                title: Some("Splitrail Analytics".to_string()),
                icons: None,
                website_url: Some("https://splitrail.dev".to_string()),
            },
            instructions: Some(
                "Splitrail MCP Server - Analytics for AI coding tools. \
                 Query daily stats, model usage, costs, file operations, and compare tools."
                    .to_string(),
            ),
        }
    }

    async fn list_resources(
        &self,
        _request: Option<PaginatedRequestParam>,
        _context: RequestContext<RoleServer>,
    ) -> Result<ListResourcesResult, McpError> {
        Ok(ListResourcesResult {
            resources: vec![
                RawResource::new(resource_uris::DAILY_SUMMARY, "Daily Summary".to_string())
                    .no_annotation(),
                RawResource::new(
                    resource_uris::MODEL_BREAKDOWN,
                    "Model Breakdown".to_string(),
                )
                .no_annotation(),
            ],
            next_cursor: None,
            meta: None,
        })
    }

    async fn read_resource(
        &self,
        ReadResourceRequestParam { uri }: ReadResourceRequestParam,
        _context: RequestContext<RoleServer>,
    ) -> Result<ReadResourceResult, McpError> {
        match uri.as_str() {
            resource_uris::DAILY_SUMMARY => {
                let stats = self.load_stats()?;
                let all_messages: Vec<_> = stats
                    .analyzer_stats
                    .iter()
                    .flat_map(|s| s.messages.iter().cloned())
                    .collect();
                let daily_stats = utils::aggregate_by_date(&all_messages);

                // Get most recent day's summary
                let summary = if let Some((date, ds)) = daily_stats.iter().next_back() {
                    format!(
                        "Date: {}\nMessages: {} user, {} AI\nConversations: {}\nCost: ${:.2}\nTokens: {} in, {} out",
                        date,
                        ds.user_messages,
                        ds.ai_messages,
                        ds.conversations,
                        ds.stats.cost(),
                        ds.stats.input_tokens,
                        ds.stats.output_tokens
                    )
                } else {
                    "No data available".to_string()
                };

                Ok(ReadResourceResult {
                    contents: vec![ResourceContents::text(summary, uri)],
                })
            }
            resource_uris::MODEL_BREAKDOWN => {
                let stats = self.load_stats()?;
                let mut model_counts: HashMap<String, u32> = HashMap::new();

                for analyzer_stats in &stats.analyzer_stats {
                    for ds in analyzer_stats.daily_stats.values() {
                        for (model, count) in &ds.models {
                            *model_counts.entry(model.clone()).or_insert(0) += count;
                        }
                    }
                }

                let mut sorted: Vec<_> = model_counts.into_iter().collect();
                sorted.sort_by_key(|b| std::cmp::Reverse(b.1));

                let breakdown: String = sorted
                    .iter()
                    .map(|(model, count)| format!("{}: {} messages", model, count))
                    .collect::<Vec<_>>()
                    .join("\n");

                let content = if breakdown.is_empty() {
                    "No model usage data available".to_string()
                } else {
                    breakdown
                };

                Ok(ReadResourceResult {
                    contents: vec![ResourceContents::text(content, uri)],
                })
            }
            _ => Err(McpError::resource_not_found(
                "resource_not_found",
                Some(rmcp::serde_json::json!({ "uri": uri })),
            )),
        }
    }
}

/// Run the MCP server with stdio transport
pub async fn run_mcp_server() -> anyhow::Result<()> {
    use rmcp::transport::stdio;

    let server = SplitrailMcpServer::new();
    let service = server.serve(stdio()).await?;
    service.waiting().await?;
    Ok(())
}

Technical Specification: Agentic Medical Device Reviewer
1. Introduction
1.1 Purpose
The Agentic Medical Device Reviewer is a sophisticated, single-page application (SPA) built using Python and Streamlit. It is designed to act as an AI-powered co-pilot for regulatory affairs professionals, specifically tailored for FDA 510(k) and Taiwan FDA (TFDA) premarket reviews. The system leverages large language models (LLMs) via an agentic pipeline, allowing users to ingest unstructured documents, extract specific regulatory features, execute specialized "skills," manage structured notes, and chain autonomous agents into complex sequential workflows.
1.2 Scope
This technical specification details the software architecture, design patterns, internal subsystems, and data workflows of the application. It provides an exhaustive breakdown of how the application routes prompts to disparate LLM providers, manages state across Streamlit’s reactive re-runs, normalizes arbitrary agent configurations into strict schemas, and renders a highly customized User Interface (UI).
1.3 Target Audience
This document is intended for software engineers, prompt engineers, system architects, and technical product managers who are responsible for maintaining, extending, or deploying the Agentic Medical Device Reviewer.
1.4 Glossary of Terms
Agent: A modular, reusable LLM configuration defined by a specific system prompt, user prompt template, model choice, temperature, and token constraints.
Skill (skill.md): A human-readable Markdown document defining a specific standard operating procedure (SOP), which the application can translate into an executable Agent.
Workflow Runner: A subsystem that chains multiple Agents in a specific sequence, feeding the output of one Agent as the input to the next.
Note Keeper: A specialized subsystem for formatting, analyzing, and interrogating regulatory notes using localized "Magics" (specialized sub-agents).
Session State: Streamlit’s in-memory dictionary (st.session_state) used to persist variables across UI interactions.
2. System Architecture
2.1 Architectural Paradigm
The application is built on a Monolithic Reactive Architecture driven by Streamlit. In this paradigm, the entire script runs from top to bottom upon every user interaction (e.g., a button click or a text input). To prevent expensive recalculations and loss of user data, the application heavily relies on decoupled state management, isolating the UI rendering logic from the state-mutation logic.
2.2 Technology Stack
Core Framework: Python 3.8+, Streamlit
Data Manipulation: Pandas
Data Visualization: Altair
Document Processing: PyPDF (PdfReader)
Configuration Management: PyYAML (yaml)
LLM Integrations:
OpenAI (openai)
Google Generative AI (google.generativeai)
Anthropic (anthropic)
X.AI / Grok (httpx for direct REST API integration)
2.3 Subsystem Overview
The system is divided into several loosely coupled layers:
UI/Theme Engine: Manages CSS injection, internationalization (i18n), and dynamic aesthetic skins.
LLM Abstraction Layer: A unified router (call_llm) that standardizes asynchronous/synchronous API calls to various LLM vendors.
Ingestion & Context Engine: Handles file uploads (PDF, TXT, MD, CSV, JSON), chunking, text extraction, and token-bounded context assembly.
Agent & YAML Engine: Parses, validates, and AI-repairs YAML definitions into a standardized schema.
Functional Modules: The core business logic segmented into seven distinct UI tabs (Dashboard, TFDA, 510(k), PDF->MD, Note Keeper, Studio, Workflow Runner).
Telemetry & Logging: A localized, session-bound logging system for debugging and auditing LLM calls and system errors.
3. Core Abstractions & Subsystems
3.1 LLM Calling Layer (call_llm)
The most critical backend component is the call_llm function. Because the application supports multiple models with divergent SDKs, call_llm serves as the universal adapter.
Parameters:
model: String identifier (e.g., gpt-4o-mini, gemini-2.5-flash).
system_prompt: The strict instructional boundary for the LLM.
user_prompt: The variable user input or chained output.
max_tokens: Bounded integer dictating response length (clamped to 120,000).
temperature: Float dictating determinism (clamped between 0.0 and 1.0).
Provider Implementations:
OpenAI: Instantiates the OpenAI client. Maps system_prompt to the "system" role and user_prompt to the "user" role. Extracts content from resp.choices[0].message.content.
Google Gemini: Instantiates genai.GenerativeModel. To ensure backward compatibility with older versions of the Google SDK that do not support the system_instruction keyword argument, the system intelligently creates a combined_prompt. It injects the system instructions at the top with a System Instruction: header, followed by the user input under a User Input: header. This fallback prevents TypeError crashes while maintaining prompt adherence.
Anthropic: Utilizes the Anthropic client. Passes the system prompt via the top-level system parameter and user prompt via the messages array. Handles the list-based block response format native to Claude models.
Grok (X.AI): Implemented via direct REST calls using httpx. Constructs a standard chat completion payload and posts to https://api.x.ai/v1/chat/completions. Handles Bearer token authorization and JSON parsing.
Error Handling & Telemetry:
Duration tracking: Records execution time in seconds.
Error capturing: Catch blocks register the exact exception type and message, updating the live_log subsystem and setting a provider_last_error flag in the session state to visually warn the user on the Dashboard.
3.2 Global State Management
Streamlit’s st.session_state is the central nervous system. The application initializes state keys in init_state() to guarantee they exist before UI rendering.
Primary State Keys:
settings: Dictionary storing theme, lang, painter_style, model, max_tokens, temperature, and allow_custom_models.
api_keys: Dictionary caching user-provided keys for OpenAI, Gemini, Anthropic, and Grok.
agents_cfg: The active, normalized dictionary containing the current schema of agents.
history: List of dictionaries recording every LLM execution (timestamp, tab, agent, model, estimated tokens).
live_log: List of dictionaries containing granular system events (INFO, WARN, ERROR).
provider_last_error: Dictionary tracking the most recent API failures per provider.
(Module-specific keys): Over 50 different keys prefix-bound to their respective modules (e.g., wf_ for Workflow Runner, note_ for Note Keeper).
3.3 Document Ingestion and Context Assembly
Medical device reviews require heavy document analysis. The app handles this via read_uploaded_file_to_text and assemble_context_from_inputs.
Extraction Strategies:
PDF: Uses pypdf.PdfReader. Wraps page extraction in a try/except loop to prevent a single corrupt page from failing the entire document. Adds [EXTRACT_ERROR] placeholders if a page fails.
CSV: Uses pandas.read_csv, converting tabular data into a compacted CSV string representation suitable for LLM consumption.
JSON: Decodes, parses, and re-dumps with indent=2 for LLM readability.
TXT/MD: Best-effort UTF-8 decoding.
Context Assembly:
To prevent exceeding context windows (which causes API rejections and financial cost spikes), the system enforces two hard limits:
MAX_DOC_CHARS_PER_FILE (120,000 characters).
MAX_TOTAL_CONTEXT_CHARS (300,000 characters).
The assemble_context_from_inputs function loops through pasted text and uploaded files. It truncates files that exceed the file limit and appends a [...TRUNCATED...] warning. It keeps a running total of characters; if the total context exceeds the global limit, the entire assembled string is truncated, and a truncated_total flag is set in the metadata to warn the user in the UI.
3.4 Agent Configuration Engine (YAML)
The core philosophy of the application is that "Agents" are merely configured states of an LLM. These states are defined in an agents.yaml file.
Standardized Schema Requirements:
code
Yaml
agents:
  <agent_id>:
    name: string
    description: string
    category: string
    model: string
    temperature: float
    max_tokens: int
    system_prompt: string
    user_prompt_template: string (must contain {{input}})
Normalization Pipeline (normalize_agents_yaml):
Because users may paste malformed YAML or non-standard schemas, the app employs a highly robust dual-stage normalization pipeline:
Parse Attempt: Tries yaml.safe_load. If it fails due to syntax errors, and use_ai_if_needed is true, it calls an LLM with a strict prompt to repair the YAML syntax without altering intent.
Structural Mapping (Heuristic): Analyzes the parsed dictionary. If it does not find the top-level agents: key, it employs deterministic best-effort mapping. It checks if the top-level is a list of agents or a dictionary of agents missing the root key, reconstructing it into the standard shape.
Field Normalization: Iterates through every agent. Normalizes IDs via Regex (^[a-z][a-z0-9_]*$). Employs a fallback dictionary (pick() function) to map synonymous keys (e.g., llm or engine becomes model; sys_prompt becomes system_prompt).
Type Coercion: Casts temperature to float (clamped 0.0-1.0) and tokens to int (clamped 256-120000).
Prompt Validation: Ensures user_prompt_template contains the placeholder {{input}}. If missing, it automatically appends \n\n{{input}} to prevent inputs from being swallowed.
AI Semantic Standardization (Optional): If the heuristic approach produces a subpar structure, it can pass the JSON representation to the LLM to rewrite it perfectly into the schema.
The engine also includes an agents_yaml_quality_score function, which deducts points for missing fields, bad IDs, and missing templates, resulting in a 0-100 score displayed on the Dashboard.
4. Functional Modules (The UI Tabs)
4.1 Dashboard
The Dashboard provides absolute operational observability over the current session.
Top-level Metrics: Calculates Total Runs, Models Used, Estimated Tokens, and Last Run Timestamp by aggregating the st.session_state["history"] dataframe.
Provider Readiness: Checks environment variables (OPENAI_API_KEY, etc.) and session state keys, displaying chips (READY or MISSING) and echoing the latest caught exception per provider.
YAML Quality Card: Renders the 0-100 quality score of the active agents_cfg.
Usage Charts (Altair):
Bar Chart 1: Runs mapped by UI Tab (identifying where the user spends the most time).
Bar Chart 2: Runs mapped by Model distribution.
Line Chart: Token consumption mapped over time, color-coded by UI Tab.
Live Log Panel: A fully filterable dataframe rendering st.session_state["live_log"]. Users can filter by module (e.g., LLM, DOCS, YAML, SKILL) and severity level (INFO, WARN, ERROR). Includes a JSON export function.
4.2 TW Premarket (TFDA)
A lightweight workspace optimized for Taiwan FDA submissions.
Design: Provides a large input area for application draft markdown.
Agent Integration: Automatically queries the active YAML for note_structurer_agent. If found, it embeds the agent_run_panel.
Routing: The output of this panel features a dedicated macro button: "Send to Workflow global prompt", which seamlessly bridges the user from draft generation into a multi-step workflow.
4.3 510(k) Intelligence
Optimized for US FDA 510(k) reviews.
Design: Requires the fda_510k_intel_agent from the YAML schema.
Execution: Accepts case inputs (product name, indications, predicate devices) and runs the predefined prompt to generate comprehensive regulatory comparison tables and risk analyses.
Routing: Features a macro button to pipe the output directly into the Note Keeper subsystem for further manipulation.
4.4 PDF to Markdown (pdf_to_markdown_agent)
A deterministic and AI-blended pipeline for Optical Character Recognition (OCR) cleanup.
Phase 1: Deterministic extraction using extract_pdf_text bounded by user-defined page ranges.
Phase 2: Agentic formatting. It uses the pdf_to_markdown_agent to take the raw, often broken OCR text (with missing line breaks and shattered tables) and reconstruct it into clean, semantic Markdown.
Routing: Features a macro button to send the structured Markdown to the "Workflow context paste" area.
4.5 AI Note Keeper & Magics
The most complex standalone text-manipulation subsystem in the application. It acts as a staging ground for unstructured thought, offering 7 distinct "Magics" (sub-agent tasks).
Base Transformation:
Takes unstructured pasted text and runs it through a custom prompt to generate the "Effective Note" (Markdown). The user can manually edit the Effective Note, and all subsequent "Magics" operate on this edited state.
The 7 Magics:
AI Formatting: Prompts the LLM to only improve readability (bolding, spacing, hierarchy) with strict instructions not to alter factual content.
AI Keywords: A hybrid deterministic/UI feature. The user inputs comma-separated keywords and picks a hex color. The system uses regex (re.sub with re.IGNORECASE) to wrap matches in HTML <span> tags with inline CSS background colors, rendering natively in Streamlit via unsafe_allow_html=True.
AI Entities: Prompts the LLM (defaulting to a fast model like gemini-2.5-flash) to extract at least 20 key entities (devices, regulations, sponsors) into a Markdown table.
AI Chat: A scoped Q&A system. It concatenates the Effective Note with a user query, instructing the LLM to answer only based on the provided context, preventing hallucinated regulatory advice.
AI Summary: Generates bulleted abstracts based on a user-defined sub-prompt.
AI Consistency Checker: A high-value regulatory feature. Prompts gpt-4o-mini to scan the text for internal logical contradictions (e.g., stating Class II in one paragraph and Class III in another). Outputs an Issue/Evidence/Risk/Fix table.
AI Citation & Traceability: Prompts the LLM to build a Claim -> Source Excerpt -> Confidence traceability matrix, highly critical for regulatory audits.
4.6 Agents & Skills Studio
A dual-purpose IDE for defining system behaviors.
Sub-Tab 1: Agents YAML
Allows uploading or pasting raw YAML.
Exposes the normalize_agents_yaml pipeline. Includes toggleable LLM-assistance for syntax repair.
Generates a comprehensive build_normalization_report_md detailing the structural adjustments made, quality score, and schema drift warnings.
Provides a text area to manually edit the resulting standardized YAML, with a strict check is_standard_agents_yaml before applying it to the live session state.
Sub-Tab 2: Skills (skill.md)
Implements the "Skill Creator" paradigm. Users upload loose text or SOPs.
standardize_skill_md_with_llm: Transforms the input into a strict SKILL.md format containing YAML frontmatter and specific sections (Purpose, Inputs, Outputs, Edge Cases).
skill_md_to_agents_yaml_with_llm: Takes the structured SKILL.md and converts it into a valid agents.yaml node, which is then passed through the normalization pipeline and injected into the active session.
Sub-Tab 3: Skill -> Task Executor
Allows a user to paste a skill.md definition, define a specific task, and supply optional document context.
execute_task_using_skill: Wraps the skill definition in a system prompt that forces the LLM to treat the Markdown document as its strict operating procedure and output contract, executing the task against the provided context.
4.7 Workflow Runner
A dynamic directed acyclic graph (DAG) executor, allowing sequential chaining of agents defined in the active YAML.
Step 1: Selection & Ordering
Reads agents_cfg and populates a multiselect box.
Maintains a custom wf_ordered list in session state. Features interactive Up (↑) and Down (↓) buttons to physically swap array indices, triggering st.rerun() to update the UI instantly.
Step 2: Global Context
Accepts a global task prompt and multi-file document uploads, processing them through assemble_context_from_inputs.
Step 3: Step-by-Step Execution
Uses a numeric stepper (wf_step_index) to navigate the ordered list of agents.
State Injection: For Step 1, the input is the Global Context + Global Prompt. For Step N, the input is automatically populated from st.session_state["wf_outputs"][Agent_{N-1}].
Renders the agent_run_panel dynamically for the active step. Output can be edited by the user before advancing to the next step, ensuring "Human-in-the-Loop" (HITL) quality control at every link in the chain.
Step 4: Export
Compiles a unified run_report.md iterating through the ordered agents and concatenating their respective final outputs for download.
5. UI Engine and UX Design
5.1 Theming System
Streamlit's default UI is overridden using arbitrary HTML/CSS injection via st.markdown(..., unsafe_allow_html=True).
CSS Architecture: Targets specific .stApp, .block-container, and div[data-baseweb="input"] DOM elements.
WOW Classes: Introduces custom CSS classes (.wow-header, .wow-card, .wow-chip, .wow-metric) to create a floating, glassmorphism UI with rounded corners (16px borders), translucent backgrounds (rgba), and backdrop filters (blur(10px)).
5.2 Painter Styles (Dynamic Skins)
The application includes 20 bespoke CSS gradient backgrounds inspired by art history (e.g., "Monet Mist", "Van Gogh Night", "Klimt Gold").
Implementation: A list of tuples maps strings to CSS linear-gradient definitions.
Reactivity: When the user selects a style from the sidebar, the apply_style function mathematically derives appropriate text colors (#F3F4F6 for dark mode, #111827 for light mode), card background opacities, and accent colors, injecting the updated CSS block upon st.rerun().
Jackpot Feature: A randomized selection button random.choice(PAINTER_STYLE_NAMES) to instantly cycle aesthetics.
5.3 Internationalization (i18n)
Implementation: A custom translation function t(en: str, zh: str) -> str.
Logic: Checks st.session_state["settings"]["lang"]. If "en", returns the first argument; if "zh-Hant", returns the second.
Coverage: Extensively applied to every label, button, placeholder, and warning across the application's 1,300+ lines.
5.4 The Generic Agent Panel (agent_run_panel)
To avoid code duplication, all agent executions are routed through a generic UI factory function.
Parameters: Accepts agent_id, configuration dict, tab_key (for state collision avoidance), and initial inputs.
Features: Automatically renders the configuration, status chips (OK, WARN, BAD), model selectors (with custom overrides), temperature sliders, prompt inspection expanders, the text area for inputs, and the "Run" button. It handles the call_llm invocation and outputs the response alongside download buttons and editable text areas.
6. Error Handling, Logging, and Metrics
6.1 Telemetry (add_log)
The application implements an internal logging registry rather than standard logging.Logger, specifically to make logs visible to the end user in the Dashboard UI.
Structure: Every log appends a dictionary: {ts: timestamp, module: str, level: str, message: str, meta: dict} to st.session_state["live_log"].
Modules: Triggers across LLM (routing), DOCS (ingestion), YAML (parsing), AGENT (execution), SKILL (generation), and UI (themes).
Token Estimation: To provide metrics without importing heavy tokenizers (like tiktoken), the app relies on a heuristic est_tokens(text) function using math.ceil(len(text) / 4), which provides a rough, universally safe ceiling estimation for dashboard metrics.
6.2 Resiliency Mechanisms
Clamp Functions: A utility clamp(n, lo, hi) is aggressively used on token limits and temperatures to prevent API rejection from users entering -1 or 5000000 into inputs.
Graceful Degradation: If an LLM provider's SDK is missing (e.g., pip install google-generativeai was forgotten), the try/except block on import sets the module to None. When call_llm attempts to invoke it, it catches the NoneType, raises a clean RuntimeError, logs it, updates the Dashboard readiness chip to MISSING, and shows the user exactly what dependency to install.
7. Security and Privacy
7.1 API Key Handling
Hierarchy: The system requests keys via os.getenv first (for secure cloud deployments). If missing, it falls back to the sidebar text inputs.
State Isolation: Keys entered via the UI are stored exclusively in st.session_state["api_keys"]. Because Streamlit state is memory-bound to the specific user session, keys are never written to disk, and a page refresh obliterates them.
Obfuscation: Text inputs for API keys use type="password" to prevent shoulder-surfing.
7.2 Data Retention
No external databases are configured by default. All documents, generated notes, and YAML changes live in memory.
The "Clear history" and "Clear live log" sidebar buttons instantly reinitialize their respective lists, effectively purging the workspace trace.
8. Extensibility and Future Work
8.1 Adding New LLM Providers
The architecture is designed for trivial provider expansion. To add a new provider (e.g., Cohere):
Add the SDK to the top try/except block.
Add model string identifiers to SUPPORTED_MODELS.
Add the env variable to PROVIDER_ENV_VARS.
Update provider_for_model heuristic.
Add a routing block inside call_llm to translate the system/user prompts into the Cohere payload structure.
8.2 Future Roadmap
Persistent Storage: Integrating SQLite or Postgres to persist agents.yaml configurations and workflow histories across sessions.
Vector Search (RAG): Currently, document context is stuffed linearly into the prompt. Future versions could integrate faiss or chromadb within assemble_context_from_inputs to perform semantic chunking and retrieval, bypassing the strict character limits.
Parallel Execution: The Workflow Runner executes sequentially. Leveraging asyncio could allow the DAG to run independent agents simultaneously.
9. Appendix
9.1 Environment Variables
To operate without manual key entry, configure the environment:
OPENAI_API_KEY: For GPT-4o variants.
GEMINI_API_KEY: For Gemini 2.5/3.0 variants.
ANTHROPIC_API_KEY: For Claude 3.5 variants.
GROK_API_KEY: For X.AI Grok variants.
9.2 Limitations & Constraints
Memory Bound: Heavy PDF uploads (e.g., 500+ pages) parsed via PyPDF will cause a RAM spike on the host machine.
Cost: Deeply chained Workflow operations with maximum token contexts can rapidly accumulate API costs, specifically on proprietary endpoints. Bounding constraints (MAX_TOTAL_CONTEXT_CHARS) are in place but do not prevent iterative looping costs by the end user.

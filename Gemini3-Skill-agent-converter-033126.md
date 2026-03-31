Comprehensive Technical Specification: WOW Regulatory Workbench
Document Version: 1.0.0
Date: March 31, 2026
Status: Finalized Design Specification
Target Audience: Software Engineers, System Architects, Product Managers, and QA Automation Engineers.
1. Executive Summary
The WOW Regulatory Workbench is a cutting-edge, AI-driven web application designed to orchestrate, manage, and execute complex regulatory compliance workflows using Large Language Models (LLMs). Built as a modern Single Page Application (SPA), it empowers compliance officers, legal analysts, and regulatory engineers to define AI "Agents" with specific "Skills," and chain them together into automated, sequential workflows.
This technical specification outlines the architectural decisions, component design, data flow, external integrations, and security considerations of the current implementation. The system leverages React 18, Vite, Tailwind CSS, and the @google/genai SDK to provide a highly responsive, client-side experience. By processing complex documents (including PDFs, CSVs, and Markdown) and feeding them through customizable agent chains, the WOW Regulatory Workbench significantly reduces the manual overhead of regulatory analysis, ensuring high accuracy and comprehensive auditability.
2. Introduction & Purpose
2.1 Problem Statement
In the highly regulated sectors of finance, healthcare, and environmental compliance, organizations struggle with the manual review of massive, dense documents. Traditional software lacks the semantic understanding required to parse regulatory nuance. While LLMs offer a solution, interacting with them via standard chat interfaces is unscalable for multi-step, deterministic regulatory workflows. Organizations need a structured environment where specific AI personas (Agents) can be configured with strict operational boundaries (Skills) and executed in a predictable, repeatable sequence.
2.2 Product Vision
The WOW Regulatory Workbench serves as an orchestration layer over the Gemini API. It abstracts the complexity of prompt engineering and context management into a visual, user-friendly dashboard. The vision is to provide a "no-code/low-code" environment where domain experts can build AI pipelines that ingest regulatory texts, analyze them against internal policies, and output structured compliance reports.
2.3 Scope of this Document
This document covers the technical implementation of the frontend application. It details the state management strategy, the component hierarchy, the document parsing mechanisms (specifically PDF extraction), the integration with the Gemini API, and the UI/UX design system. It does not cover backend database schemas or server-side API design, as the current iteration operates entirely within the client browser, utilizing browser memory for state and direct client-to-API communication for LLM inference.
3. System Architecture
3.1 High-Level Architecture
The application follows a Client-Side Rendering (CSR) architecture. It is built as a Single Page Application (SPA) to ensure fluid transitions between different modules (Dashboard, Config, Skills, Workflow) without page reloads.
The architecture is composed of three primary layers:
Presentation Layer (UI): React components styled with Tailwind CSS and shadcn/ui.
State Management Layer: React Hooks (useState, useEffect) managing the global application state at the root component level.
Integration Layer: Utility functions and SDKs handling external communication (Gemini API via @google/genai) and complex browser-side processing (PDF parsing via pdfjs-dist).
3.2 Technology Stack Justification
Core Framework: React 18. Chosen for its robust ecosystem, component-based architecture, and concurrent rendering capabilities which ensure the UI remains responsive during heavy local processing (like PDF text extraction).
Build Tool: Vite. Selected over Webpack or Create React App for its native ES modules support, resulting in near-instantaneous Hot Module Replacement (HMR) and significantly faster build times.
Language: TypeScript (ES2022 target). Provides static typing, enhancing code maintainability, reducing runtime errors, and providing superior developer ergonomics via IDE autocompletion.
Styling: Tailwind CSS. A utility-first CSS framework that allows for rapid UI development without leaving the markup. It ensures a consistent design system and minimizes the CSS bundle size through PurgeCSS (integrated into Tailwind's JIT compiler).
Component Library: shadcn/ui. Provides accessible, unstyled, and customizable UI components (Tabs, Cards, Toasters). Unlike traditional component libraries, shadcn/ui components are copied directly into the codebase, offering complete control over the markup and styling.
Icons: Lucide React. A clean, consistent, and lightweight SVG icon library.
AI Integration: @google/genai SDK. The official Google GenAI SDK for interacting with Gemini models (Gemini 3.1 Pro, Gemini 3 Flash).
Document Processing: pdfjs-dist (v5.6+). Mozilla's robust PDF parsing library, utilized via Web Workers to extract text from PDF documents without blocking the main UI thread.
3.3 Component Hierarchy
The application follows a strict hierarchical tree structure, ensuring unidirectional data flow (top-down) and event bubbling (bottom-up via callback functions).
code
Text
App (Root)
├── Header (Static UI)
├── Main Content Area
│   ├── Tabs Navigation
│   │   ├── Dashboard (Displays metrics)
│   │   ├── AgentsConfigStudio (YAML Editor)
│   │   ├── SkillCreator (Markdown Editor)
│   │   └── WorkflowRunner (Execution Engine)
│   └── LiveLog (Sidebar Event Stream)
└── Toaster (Global Notification System)
4. Core Modules & Component Design
This section provides a deep dive into the specific responsibilities, internal state, and logic of each major module within the application.
4.1 App Shell & Global State (App.tsx)
The App component acts as the central nervous system of the application. Because the current architecture does not utilize a global state management library like Redux or Zustand, the App component holds the primary state variables and passes them down as props to the child modules.
Global State Variables:
agentsYaml (String): The raw YAML string representing the configured agents. This is the single source of truth for agent definitions across the app.
liveLog (Array of Objects): A chronological array of system events. Each object contains a timestamp, a message, and a type ('info', 'success', 'error', 'warning').
runHistory (Array of Objects): A historical record of all executed agent workflows. Used primarily by the Dashboard to calculate metrics.
Global Methods:
addLog(message, type): A callback function passed to all child components. It appends a new log entry to the liveLog state, triggering a re-render of the LiveLog sidebar.
addRun(run): A callback function passed to the WorkflowRunner. It appends a completed execution record to the runHistory state.
4.2 Dashboard Module (Dashboard.tsx)
The Dashboard provides a high-level, analytical overview of the user's session. It is a purely presentational component that derives its UI from the runHistory prop.
Key Features & Calculations:
Total Runs: A simple length calculation of the runHistory array.
Success Rate: Calculates the percentage of runs where status === 'success'. It handles division-by-zero edge cases gracefully.
Average Execution Time: Iterates through the runHistory, summing the duration property, and dividing by the total runs. Formatted to one decimal place.
Total Tokens: Aggregates the estimated token usage across all runs.
Recent Activity Feed: Displays the 5 most recent runs in reverse chronological order, utilizing color-coded indicators (emerald for success, red for failure) and displaying the agent name, timestamp, and duration.
4.3 Agents Config Studio (AgentsConfigStudio.tsx)
This module is responsible for managing the declarative configuration of the AI agents. It provides a raw text editing interface for YAML.
Functional Requirements:
State Binding: The textarea is fully controlled, bound to the global agentsYaml state.
AI Standardization: The core feature is the "Standardize YAML" button. When clicked, it takes the user's potentially malformed or incomplete YAML and sends it to the Gemini API with a strict system instruction to format it according to the application's expected schema (requiring fields like name, role, goal, backstory, and skills).
Error Handling: If the Gemini API fails to return valid YAML, the component catches the error, alerts the user via the Toaster, and logs the failure in the LiveLog.
Export: Allows users to download the current YAML state as a .yaml file using a dynamically generated Blob and an invisible anchor (<a>) tag.
4.4 Skill Creator (SkillCreator.tsx)
The Skill Creator is a specialized module for defining the operational boundaries and instructions for agents. Skills are authored in Markdown.
Functional Requirements:
Local State: Maintains its own rawSkill (the user's input) and standardizedSkill (the AI-processed output) state.
AI Transformation: Uses the Gemini API to transform unstructured text into a highly structured Markdown document. The prompt engineering here is critical: it instructs the LLM to extract the core intent and format it with specific headers (Skill Name, Description, Prerequisites, Execution Steps, Expected Output).
Integration with Agents: The "Transform to agents.yaml" feature is a complex state mutation. It takes the standardized skill, uses the Gemini API to generate a new Agent YAML block that incorporates this skill, and appends this new block to the global agentsYaml state. This provides a seamless transition from defining a capability to instantiating an agent that possesses it.
4.5 Workflow Runner (WorkflowRunner.tsx)
The Workflow Runner is the most technically complex module in the application. It acts as the execution engine, handling file parsing, agent chaining, and asynchronous API communication.
4.5.1 Agent Parsing & Selection
The component parses the global agentsYaml string into a JavaScript object. It renders a UI allowing users to select multiple agents to form an execution chain. The order of selection determines the execution sequence.
4.5.2 Document Ingestion & Parsing
The runner supports multiple file types for context ingestion: .txt, .md, .csv, and .pdf.
Text-based files: Read using the native browser FileReader API (readAsText).
PDF files: This requires heavy lifting. The application utilizes pdfjs-dist.
Worker Initialization: The PDF.js worker is initialized pointing to a CDN (cdnjs.cloudflare.com) to offload the parsing logic from the main UI thread.
ArrayBuffer Conversion: The uploaded file is converted to an ArrayBuffer.
Page Iteration: The pdfjsLib.getDocument method is called. The code then iterates through every page (pdf.numPages), extracts the text content (page.getTextContent()), and concatenates the string items.
Memory Management: Care is taken to ensure the promise chain resolves correctly and memory is freed after extraction.
4.5.3 The Execution Engine (The Chain)
When the user clicks "Run Workflow", a complex asynchronous state machine is triggered:
Initialization: The isRunning state is set to true. The results array is cleared.
Context Setup: The initial context is set to the parsed document text (if uploaded) combined with any manual context provided by the user.
Sequential Iteration: A for...of loop iterates through the selected agents.
Prompt Construction: For each agent, a dynamic prompt is constructed. This prompt includes:
The Agent's Role, Goal, and Backstory (parsed from the YAML).
The user-defined task for this specific step in the chain.
The Context: Crucially, the context passed to Agent N is the output generated by Agent N-1. This creates the "chain" effect.
API Invocation: The @google/genai SDK is called using ai.models.generateContent. The user can select the model (e.g., gemini-3.1-pro-preview for complex reasoning, gemini-3-flash-preview for speed).
Result Processing: The response text is extracted. The execution time is calculated. The result is appended to the local results state (for UI display) and the global runHistory state (for the Dashboard).
Completion: Once all agents have executed, isRunning is set to false, and a success notification is dispatched.
4.6 Live Log System (LiveLog.tsx)
The LiveLog provides real-time observability into the system's state.
Functional Requirements:
Prop Drilling: Receives the logs array from the App component.
Auto-Scrolling: Utilizes a useRef hook attached to an invisible div at the bottom of the log list. A useEffect hook triggers scrollIntoView({ behavior: 'smooth' }) whenever the logs array changes, ensuring the user always sees the most recent event.
Visual Hierarchy: Uses Tailwind utility classes to color-code log entries based on their type (e.g., text-emerald-500 for success, text-red-500 for errors), allowing users to quickly scan for issues.
5. Data Models & State Management
5.1 State Definitions
The application relies on specific data structures to maintain consistency.
LiveLog Entry Interface:
code
TypeScript
interface LogEntry {
  timestamp: Date;
  message: string;
  type: 'info' | 'success' | 'error' | 'warning';
}
Run History Entry Interface:
code
TypeScript
interface RunRecord {
  agentName: string;
  timestamp: Date;
  duration: number; // in seconds
  status: 'success' | 'error';
  tokens?: number; // Estimated token count
}
Agent Configuration Schema (Expected YAML Structure):
While stored as a string, the system expects the YAML to conform to the following conceptual schema:
code
Yaml
agents:
  - name: string
    role: string
    goal: string
    backstory: string
    skills:
      - string
5.2 Data Flow Architecture
The data flow is strictly unidirectional.
User Input: The user interacts with a child component (e.g., types in the YAML editor, uploads a file).
Local State Update: The child component updates its local state to reflect the immediate UI change.
Global Action: If the action has global implications (e.g., running a workflow), the child component invokes a callback passed down from App.tsx (e.g., addRun, addLog).
Global State Mutation: App.tsx updates its state (runHistory, liveLog).
Re-render: React propagates the new state down to all relevant child components (e.g., the Dashboard receives the new runHistory and updates its charts/metrics).
6. External Integrations
6.1 Gemini API Integration (@google/genai)
The application relies heavily on the Gemini API for its core functionality.
Initialization:
The SDK is initialized using the standard pattern:
code
TypeScript
import { GoogleGenAI } from "@google/genai";
const ai = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY });
(Note: In the actual implementation, the API key is injected by the AI Studio environment, but the architectural principle remains the same).
Model Selection Strategy:
Standardization Tasks (Agents/Skills): The application defaults to gemini-3-flash-preview. These tasks require fast text formatting and structural manipulation rather than deep, complex reasoning. Using Flash minimizes latency for UI interactions.
Workflow Execution: The user is given the choice between gemini-3.1-pro-preview and gemini-3-flash-preview. Pro is recommended for deep regulatory analysis, legal text comprehension, and complex multi-step reasoning. Flash is provided as an option for rapid prototyping or simpler tasks.
Prompt Engineering Paradigms:
The application uses "System Instructions" embedded within the prompt to enforce output formats. For example, when standardizing YAML, the prompt explicitly states: "You are an expert AI configuration engineer... Return ONLY valid YAML. Do not include markdown formatting blocks." This strict prompting reduces parsing errors on the client side.
6.2 PDF Parsing (pdfjs-dist)
Integrating PDF parsing in a client-side React application presents unique challenges, primarily regarding bundle size and main-thread blocking.
Implementation Details:
Worker Thread: To prevent the UI from freezing while parsing large regulatory PDFs (which can be hundreds of pages), pdfjs-dist is configured to use a Web Worker.
CDN Sourcing: The worker script is sourced from a CDN (cdnjs) rather than bundled directly. This significantly reduces the initial load time of the application.
code
TypeScript
pdfjsLib.GlobalWorkerOptions.workerSrc = `//cdnjs.cloudflare.com/ajax/libs/pdf.js/${pdfjsLib.version}/pdf.worker.min.mjs`;
Text Extraction Algorithm: The application does not attempt to render the PDF visually. It strictly extracts the text layer. It iterates through each page, extracts the TextContent items, and joins them with spaces. This provides a raw string suitable for LLM context window ingestion.
7. User Interface & User Experience (UI/UX)
7.1 Design System (Tailwind CSS)
The application utilizes a custom design system built on top of Tailwind CSS, focusing on a clean, professional, "enterprise software" aesthetic.
Color Palette: Heavily relies on the slate color palette for backgrounds (bg-slate-50), borders (border-slate-200), and typography (text-slate-900, text-slate-500). This creates a low-contrast, easy-on-the-eyes environment suitable for long sessions of reading regulatory text.
Accents: indigo-600 is used for primary brand elements and active states. emerald-500 is used consistently across the app to indicate success or "System Online" status. red-500 is reserved strictly for errors.
Typography: Utilizes the default sans-serif stack (font-sans), ensuring native, highly readable text rendering across different operating systems. Tracking is tightened (tracking-tight) on headers for a modern look.
7.2 Component Library (shadcn/ui)
The integration of shadcn/ui provides robust, accessible primitives.
Tabs: Used for the primary navigation. The active state is styled with a white background and shadow to mimic a physical toggle, providing clear spatial orientation.
Cards: Used extensively in the Dashboard and Workflow Runner to group related information and create visual hierarchy.
Toaster (Sonner): Provides non-intrusive, transient feedback for user actions (e.g., "YAML Standardized Successfully", "Workflow Completed"). Positioned at the bottom-right to avoid obscuring primary content.
7.3 Layout & Responsiveness
Grid System: The main layout utilizes CSS Grid (grid-cols-1 lg:grid-cols-4). On large screens, the main content takes up 3 columns, and the LiveLog takes up 1 column. On smaller screens, they stack vertically.
Container Constraints: The main content is constrained by max-w-7xl and centered using mx-auto, preventing the UI from stretching uncomfortably on ultra-wide monitors.
8. Security & Compliance Considerations
While the current application is a client-side prototype, several security principles are addressed, and others are identified for future production deployment.
8.1 API Key Management
Current State: In the AI Studio environment, the Gemini API key is injected securely into the runtime.
Production Consideration: For a true production deployment, storing the Gemini API key in the client-side bundle (even as an environment variable) is a critical security risk. A malicious user could extract the key and abuse the quota.
Required Architecture Change: The application must transition to a Full-Stack architecture. The frontend should make requests to a custom backend (e.g., Node.js/Express), which securely holds the API key, authenticates the user, and proxies the request to the Gemini API.
8.2 Data Privacy & Handling
Current State: All document parsing (including sensitive regulatory PDFs) happens entirely locally within the user's browser using pdfjs-dist. The document text is only transmitted over the network when it is sent to the Gemini API as part of a prompt.
Compliance Note: Organizations must ensure that their agreement with Google Cloud (regarding the Gemini API) complies with their internal data privacy policies (e.g., ensuring data sent to the API is not used to train foundational models).
8.3 Input Sanitization
Current State: The application relies on React's built-in XSS protection when rendering text. However, when parsing raw YAML or Markdown, there is a theoretical risk of injection if the data is subsequently used in an unsafe manner (e.g., dangerouslySetInnerHTML).
Mitigation: The application avoids dangerouslySetInnerHTML. All outputs from the LLM are treated as plain text or safely rendered markdown.
9. Performance Optimization
9.1 Bundle Size Management
Tree Shaking: Vite and Rollup automatically tree-shake unused exports from libraries like lucide-react, ensuring only the imported icons are included in the final bundle.
Dynamic Imports (Future Optimization): Currently, pdfjs-dist is imported statically. For faster initial load times, this heavy library could be dynamically imported (await import('pdfjs-dist')) only when the user navigates to the Workflow Runner and attempts to upload a PDF.
9.2 React Rendering Optimization
State Colocation: State is kept as close to where it is needed as possible. For example, the rawSkill state is local to the SkillCreator component. Typing in the skill editor does not trigger a re-render of the entire App component, only the SkillCreator.
List Rendering: When rendering the runHistory or liveLog, React key props are used (currently using index i, but should ideally use unique IDs in the future) to help React identify which items have changed, been added, or been removed, optimizing the reconciliation process.
10. Deployment & DevOps Strategy
10.1 Build Process
The application is built using the standard Vite build pipeline:
code
Bash
npm run build
This command triggers TypeScript compilation (via tsc) and bundling (via Vite/Rollup). The output is a highly optimized, minified set of static HTML, CSS, and JavaScript files placed in the dist/ directory.
10.2 Hosting Environment
Because the application is currently a purely static Client-Side SPA, it can be hosted on any static file server or CDN.
Recommended Platforms: Vercel, Netlify, AWS S3 + CloudFront, or Google Cloud Storage + Cloud CDN.
Routing: The hosting provider must be configured to redirect all 404 requests to index.html to allow React Router (if implemented in the future) to handle client-side routing.
10.3 CI/CD Pipeline Recommendations
For a production environment, a CI/CD pipeline (e.g., GitHub Actions, GitLab CI) should be implemented with the following stages:
Linting: Run ESLint to enforce code style and catch potential errors.
Type Checking: Run tsc --noEmit to ensure TypeScript compilation succeeds.
Testing: Run unit tests (e.g., Vitest) and component tests (e.g., React Testing Library).
Build: Execute npm run build.
Deploy: Push the dist/ artifacts to the hosting environment.
11. Future Roadmap & Extensibility
The current implementation serves as a powerful foundational prototype. To evolve into an enterprise-grade platform, the following architectural enhancements are recommended for Phase 2:
11.1 Backend Integration & Database
Transition from a purely client-side state to a persistent backend.
Database: Implement PostgreSQL or MongoDB to store user profiles, agent configurations (agents.yaml equivalents), and historical run logs.
API Layer: Develop a Node.js/Express or Python/FastAPI backend to handle database operations, secure API key management, and proxy requests to the Gemini API.
11.2 Authentication & Role-Based Access Control (RBAC)
Integrate an identity provider (e.g., Auth0, Firebase Auth, or enterprise SSO via SAML/OAuth2).
Implement RBAC to restrict access to sensitive regulatory agents or specific document types based on user roles (e.g., Admin, Analyst, Auditor).
11.3 Advanced Workflow Capabilities
Visual Node Editor: Replace the linear agent selection UI with a visual, drag-and-drop node editor (using libraries like React Flow) to support complex branching logic, conditional execution, and parallel agent processing.
Human-in-the-Loop (HITL): Introduce pause states in the workflow where a human user must review, edit, or approve an agent's output before the workflow proceeds to the next step.
11.4 Enhanced Document Processing
OCR Integration: Integrate Optical Character Recognition (OCR) capabilities (e.g., Google Cloud Vision API) to extract text from scanned PDFs or images that do not contain a native text layer.
Vector Database & RAG: Instead of passing entire documents into the context window (which may exceed token limits), implement Retrieval-Augmented Generation (RAG). Documents would be chunked, embedded, and stored in a vector database (e.g., Pinecone, Weaviate). Agents would then query the vector database to retrieve only the most relevant document sections for their specific task.
12. Conclusion
The WOW Regulatory Workbench demonstrates a sophisticated application of modern web technologies and Large Language Models to solve complex, document-heavy compliance challenges. By utilizing React, Vite, and Tailwind CSS, the application delivers a highly responsive and intuitive user experience. The integration of pdfjs-dist for local document parsing ensures data privacy, while the strategic use of the @google/genai SDK empowers users to orchestrate powerful AI workflows without writing code. This technical specification provides a comprehensive blueprint of the current system, establishing a solid foundation for future enterprise scaling, backend integration, and advanced AI feature development.

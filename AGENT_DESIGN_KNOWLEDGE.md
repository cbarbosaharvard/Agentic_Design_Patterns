# Agentic Design Patterns: Level 2 & Level 3 Knowledge Guide

A comprehensive guide to designing, creating, implementing, and using AI agents at Level 2 and Level 3 capabilities, based on practical code examples from the Agentic Design Patterns repository.

---

## Table of Contents

1. [Agent Capability Levels Overview](#agent-capability-levels-overview)
2. [Level 2: Intermediate Agent Capabilities](#level-2-intermediate-agent-capabilities)
   - [Tool Use](#1-tool-use)
   - [Planning](#2-planning)
   - [Reflection](#3-reflection)
   - [Routing](#4-routing)
3. [Level 3: Advanced Agent Capabilities](#level-3-advanced-agent-capabilities)
   - [Multi-Agent Collaboration](#1-multi-agent-collaboration)
   - [Memory Management](#2-memory-management)
   - [Adaptation](#3-adaptation)
   - [Inter-Agent Communication (A2A)](#4-inter-agent-communication-a2a)
4. [Production Patterns](#production-patterns)
   - [Guardrails & Safety](#guardrails--safety)
   - [Human-in-the-Loop](#human-in-the-loop)
   - [Goal Setting & Monitoring](#goal-setting--monitoring)
5. [Framework Comparison](#framework-comparison)
6. [Best Practices](#best-practices)
7. [Quick Reference](#quick-reference)

---

## Agent Capability Levels Overview

| Level | Capabilities | Characteristics |
|-------|-------------|-----------------|
| **Level 1** | Prompt Chaining, Basic Parallelization | Single LLM calls, no external actions |
| **Level 2** | Tool Use, Planning, Reflection, Routing | Agents can interact with external systems and self-improve |
| **Level 3** | Multi-Agent, Memory, Adaptation, A2A | Multiple coordinated agents with persistent state |

---

## Level 2: Intermediate Agent Capabilities

Level 2 agents can autonomously interact with external tools, plan multi-step solutions, evaluate their own outputs, and route requests intelligently.

### 1. Tool Use

**Definition**: Agents that can call external functions, APIs, or execute code to accomplish tasks.

**Key Concepts**:
- **Function Tools**: Python functions exposed to the LLM
- **Tool Selection**: LLM decides which tool to use based on the task
- **Tool Execution**: Runtime executes the selected tool and returns results

#### LangChain Implementation

```python
import asyncio
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain.agents import create_tool_calling_agent, AgentExecutor

# Initialize LLM with tool-calling capabilities
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)

# Define a Tool using the @tool decorator
@tool
def search_information(query: str) -> str:
    """
    Provides factual information on a given topic. Use this tool to find answers
    to phrases like 'capital of France' or 'weather in London?'.
    """
    # Tool implementation (simulated here)
    simulated_results = {
        "weather in london": "The weather in London is currently cloudy with a temperature of 15°C.",
        "capital of france": "The capital of France is Paris.",
        "default": f"Simulated search result for '{query}'"
    }
    return simulated_results.get(query.lower(), simulated_results["default"])

tools = [search_information]

# Create Agent with Tools
agent_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),  # Required for agent's internal reasoning
])

# Bind LLM, tools, and prompt together
agent = create_tool_calling_agent(llm, tools, agent_prompt)

# AgentExecutor runs the agent and executes tools
agent_executor = AgentExecutor(agent=agent, verbose=True, tools=tools)

# Execute
async def run_agent(query: str):
    response = await agent_executor.ainvoke({"input": query})
    print(response["output"])

asyncio.run(run_agent("What is the capital of France?"))
```

#### Google ADK Implementation

```python
from google.adk.agents import Agent
from google.adk.tools import FunctionTool

# Define tool function
def booking_handler(request: str) -> str:
    """Handles booking requests for flights and hotels."""
    return f"Booking action for '{request}' has been simulated."

# Create tool from function
booking_tool = FunctionTool(booking_handler)

# Create agent with tools
booking_agent = Agent(
    name="Booker",
    model="gemini-2.0-flash",
    description="Handles all flight and hotel booking requests.",
    tools=[booking_tool]
)
```

**Best Practices for Tool Use**:
- Write clear, descriptive docstrings - the LLM uses these to understand when to call the tool
- Keep tools focused on single responsibilities
- Handle errors gracefully within tools
- Return structured data when possible

---

### 2. Planning

**Definition**: Agents that decompose complex goals into actionable sub-steps before execution.

**Key Concepts**:
- **Goal Decomposition**: Breaking down high-level objectives
- **Plan-then-Execute**: Generate plan first, then follow it
- **Iterative Refinement**: Adjust plan based on feedback

#### CrewAI Implementation

```python
from crewai import Agent, Task, Crew, Process
from langchain_openai import ChatOpenAI

# Initialize LLM
llm = ChatOpenAI(model="gpt-4-turbo")

# Define a Planning Agent
planner_writer_agent = Agent(
    role='Article Planner and Writer',
    goal='Plan and then write a concise, engaging summary on a specified topic.',
    backstory=(
        'You are an expert technical writer and content strategist. '
        'Your strength lies in creating a clear, actionable plan before writing, '
        'ensuring the final summary is both informative and easy to digest.'
    ),
    verbose=True,
    allow_delegation=False,
    llm=llm
)

# Define task with planning structure
topic = "The importance of Reinforcement Learning in AI"
planning_task = Task(
    description=(
        f"1. Create a bullet-point plan for a summary on the topic: '{topic}'.\n"
        f"2. Write the summary based on your plan, keeping it around 200 words."
    ),
    expected_output=(
        "A final report containing two distinct sections:\n\n"
        "### Plan\n"
        "- A bulleted list outlining the main points of the summary.\n\n"
        "### Summary\n"
        "- A concise and well-structured summary of the topic."
    ),
    agent=planner_writer_agent,
)

# Create and run crew
crew = Crew(
    agents=[planner_writer_agent],
    tasks=[planning_task],
    process=Process.sequential,
)

result = crew.kickoff()
print(result)
```

**Planning Patterns**:
1. **Linear Planning**: Steps executed in sequence
2. **Hierarchical Planning**: High-level plan decomposed into sub-plans
3. **Adaptive Planning**: Plan adjusts based on execution results

---

### 3. Reflection

**Definition**: Agents that evaluate and iteratively improve their own outputs.

**Key Concepts**:
- **Generate-Critique-Refine Cycle**: Three-phase improvement loop
- **Self-Evaluation**: Agent assesses its own output quality
- **Iterative Improvement**: Multiple passes to enhance quality

#### LangChain Reflection Chain

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

# Step 1: Initial Generation
generation_chain = (
    ChatPromptTemplate.from_messages([
        ("system", "Write a short, simple product description for a new smart coffee mug."),
        ("user", "{product_details}")
    ])
    | llm
    | StrOutputParser()
)

# Step 2: Critique - Evaluate the generated content
critique_chain = (
    ChatPromptTemplate.from_messages([
        ("system", """Critique the following product description based on clarity,
        conciseness, and appeal. Provide specific suggestions for improvement."""),
        ("user", "Product Description to Critique:\n{initial_description}")
    ])
    | llm
    | StrOutputParser()
)

# Step 3: Refinement - Improve based on critique
refinement_chain = (
    ChatPromptTemplate.from_messages([
        ("system", """Based on the original product details and the following critique,
        rewrite the product description to be more effective.

        Original Product Details: {product_details}
        Critique: {critique}

        Refined Product Description:"""),
        ("user", "")
    ])
    | llm
    | StrOutputParser()
)

# Build Full Reflection Chain
full_reflection_chain = (
    RunnablePassthrough.assign(initial_description=generation_chain)
    | RunnablePassthrough.assign(critique=critique_chain)
    | refinement_chain
)

# Execute
product_details = "A mug that keeps coffee hot and can be controlled by a smartphone app."
result = full_reflection_chain.invoke({"product_details": product_details})
print(result)
```

**Reflection Flow**:
```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  Generate   │ ──▶ │  Critique   │ ──▶ │   Refine    │
│  (Draft)    │     │  (Evaluate) │     │  (Improve)  │
└─────────────┘     └─────────────┘     └─────────────┘
       │                                       │
       └───────────── Loop if needed ──────────┘
```

---

### 4. Routing

**Definition**: Agents that analyze requests and delegate them to appropriate specialized handlers.

**Key Concepts**:
- **Request Classification**: Understanding intent and type
- **Dynamic Delegation**: Routing to the right specialist
- **Coordinator Pattern**: Parent agent manages sub-agents

#### Google ADK Routing Implementation

```python
import uuid
from google.adk.agents import Agent
from google.adk.runners import InMemoryRunner
from google.adk.tools import FunctionTool
from google.genai import types

# Define specialized tool functions
def booking_handler(request: str) -> str:
    """Handles booking requests for flights and hotels."""
    return f"Booking action for '{request}' has been simulated."

def info_handler(request: str) -> str:
    """Handles general information requests."""
    return f"Information request for '{request}'. Result: Simulated retrieval."

# Create tools
booking_tool = FunctionTool(booking_handler)
info_tool = FunctionTool(info_handler)

# Define specialized sub-agents
booking_agent = Agent(
    name="Booker",
    model="gemini-2.0-flash",
    description="Handles all flight and hotel booking requests.",
    tools=[booking_tool]
)

info_agent = Agent(
    name="Info",
    model="gemini-2.0-flash",
    description="Provides general information and answers questions.",
    tools=[info_tool]
)

# Define coordinator (router) agent
coordinator = Agent(
    name="Coordinator",
    model="gemini-2.0-flash",
    instruction=(
        "You are the main coordinator. Analyze incoming requests "
        "and delegate them to the appropriate specialist agent.\n"
        "- Booking requests → delegate to 'Booker' agent.\n"
        "- Information questions → delegate to 'Info' agent."
    ),
    description="Routes user requests to the correct specialist agent.",
    sub_agents=[booking_agent, info_agent]  # Enables auto-delegation
)

# Execute
def run_coordinator(runner: InMemoryRunner, request: str):
    user_id = "user_123"
    session_id = str(uuid.uuid4())
    runner.session_service.create_session(
        app_name=runner.app_name, user_id=user_id, session_id=session_id
    )

    for event in runner.run(
        user_id=user_id,
        session_id=session_id,
        new_message=types.Content(role='user', parts=[types.Part(text=request)])
    ):
        if event.is_final_response() and event.content:
            return event.content.text
    return None

runner = InMemoryRunner(coordinator)
result = run_coordinator(runner, "Book me a hotel in Paris.")
print(result)
```

**Routing Architecture**:
```
                    ┌─────────────────┐
                    │   Coordinator   │
                    │    (Router)     │
                    └────────┬────────┘
                             │
           ┌─────────────────┼─────────────────┐
           ▼                 ▼                 ▼
    ┌─────────────┐   ┌─────────────┐   ┌─────────────┐
    │   Booking   │   │    Info     │   │   Support   │
    │   Agent     │   │   Agent     │   │   Agent     │
    └─────────────┘   └─────────────┘   └─────────────┘
```

---

## Level 3: Advanced Agent Capabilities

Level 3 agents operate as coordinated systems with persistent memory, self-improvement, and inter-agent communication.

### 1. Multi-Agent Collaboration

**Definition**: Multiple specialized agents working together toward common goals.

**Collaboration Patterns**:

| Pattern | Description | Use Case |
|---------|-------------|----------|
| **Sequential** | Agents execute in order | Pipeline processing |
| **Parallel** | Agents execute simultaneously | Independent tasks |
| **Hierarchical** | Parent delegates to children | Complex orchestration |
| **Loop** | Agents iterate until condition met | Refinement tasks |

#### Google ADK Hierarchical Coordination

```python
from google.adk.agents import LlmAgent, BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event
from typing import AsyncGenerator

# Custom non-LLM agent
class TaskExecutor(BaseAgent):
    """A specialized agent with custom, non-LLM behavior."""
    name: str = "TaskExecutor"
    description: str = "Executes a predefined task."

    async def _run_async_impl(self, context: InvocationContext) -> AsyncGenerator[Event, None]:
        # Custom implementation logic
        yield Event(author=self.name, content="Task finished successfully.")

# LLM-based agent
greeter = LlmAgent(
    name="Greeter",
    model="gemini-2.0-flash-exp",
    instruction="You are a friendly greeter."
)

task_doer = TaskExecutor()

# Parent coordinator with sub-agents
coordinator = LlmAgent(
    name="Coordinator",
    model="gemini-2.0-flash-exp",
    description="A coordinator that can greet users and execute tasks.",
    instruction=(
        "When asked to greet, delegate to the Greeter. "
        "When asked to perform a task, delegate to the TaskExecutor."
    ),
    sub_agents=[greeter, task_doer]  # Establishes parent-child relationships
)

# Framework automatically manages:
# - greeter.parent_agent == coordinator
# - task_doer.parent_agent == coordinator
```

#### CrewAI Multi-Agent Crew

```python
from crewai import Agent, Task, Crew, Process
from crewai_tools import SerperDevTool

search_tool = SerperDevTool()

# Define specialized agents
researcher = Agent(
    role='Senior Research Analyst',
    goal='Provide accurate research summaries.',
    backstory='Expert in extracting insights from information.',
    tools=[search_tool],
    verbose=True
)

data_analyst = Agent(
    role='Data Analyst',
    goal='Process and validate structured data.',
    backstory='Meticulous in handling datasets.',
    tools=[],  # Restricted tools for security
    verbose=True
)

# Define tasks with dependencies
research_task = Task(
    description="Research climate change impacts on coastal cities.",
    agent=researcher,
    output_file='research_summary.json'
)

analysis_task = Task(
    description="Analyze the research summary and validate its structure.",
    agent=data_analyst,
    context=[research_task]  # Uses output from research_task
)

# Create crew
crew = Crew(
    agents=[researcher, data_analyst],
    tasks=[research_task, analysis_task],
    process=Process.sequential,
    verbose=True
)

result = crew.kickoff()
```

---

### 2. Memory Management

**Definition**: Agents that maintain and retrieve context across conversations and sessions.

**Memory Types**:

| Type | Scope | Use Case |
|------|-------|----------|
| **Short-term** | Single conversation | Context within session |
| **Long-term** | Across sessions | User preferences, history |
| **Episodic** | Specific events | Past interactions |
| **Semantic** | Factual knowledge | Domain information |

#### LangChain Memory Implementation

```python
from langchain.memory import ChatMessageHistory, ConversationBufferMemory
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

# Simple Message History
history = ChatMessageHistory()
history.add_user_message("I'm heading to New York next week.")
history.add_ai_message("Great! It's a fantastic city.")
print(history.messages)

# Conversation Buffer Memory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Build Memory-Enabled Chain
llm = ChatOpenAI()
prompt = ChatPromptTemplate(
    messages=[
        SystemMessagePromptTemplate.from_template("You are a friendly assistant."),
        MessagesPlaceholder(variable_name="chat_history"),  # Memory placeholder
        HumanMessagePromptTemplate.from_template("{question}")
    ]
)

conversation = LLMChain(llm=llm, prompt=prompt, memory=memory)

# Conversation with memory
response = conversation.predict(question="Hi, I'm Jane.")
response = conversation.predict(question="Do you remember my name?")  # Should remember "Jane"
```

#### LangGraph Store-Based Memory

```python
from langgraph.store.memory import InMemoryStore

# Initialize store with embedding support
def embed(texts: list[str]) -> list[list[float]]:
    # Use a proper embedding model in production
    return [[1.0, 2.0] for _ in texts]

store = InMemoryStore(index={"embed": embed, "dims": 2})

# Define namespace for organization
user_id = "my-user"
application_context = "chitchat"
namespace = (user_id, application_context)

# Store a memory
store.put(
    namespace,
    "user-preferences",
    {
        "rules": [
            "User likes short, direct language",
            "User only speaks English & Python",
        ],
        "custom-key": "custom-value",
    },
)

# Retrieve memory
item = store.get(namespace, "user-preferences")

# Search with filtering and vector similarity
items = store.search(
    namespace,
    filter={"custom-key": "custom-value"},
    query="language preferences"
)
```

#### ADK Session and Memory Services

```python
from google.adk.runners import Runner
from google.adk.services import (
    InMemorySessionService,
    InMemoryMemoryService,
    InMemoryArtifactService
)

# Configure runner with memory services
runner = Runner(
    app_name="MyAgent",
    agent=my_agent,
    artifact_service=InMemoryArtifactService(),
    session_service=InMemorySessionService(),  # Session-scoped memory
    memory_service=InMemoryMemoryService(),    # Long-term memory
)
```

---

### 3. Adaptation

**Definition**: Agents that learn and improve their behavior over time.

**Adaptation Mechanisms**:
- **Feedback Integration**: Learning from user corrections
- **Goal-Based Iteration**: Refining until objectives are met
- **Evolutionary Improvement**: Using techniques like OpenEvolve

#### OpenEvolve Framework

```python
from openevolve import OpenEvolve

# Initialize evolutionary system
evolve = OpenEvolve(
    initial_program_path="path/to/initial_program.py",
    evaluation_file="path/to/evaluator.py",
    config_path="path/to/config.yaml"
)

# Run evolution
best_program = await evolve.run(iterations=1000)
print(f"Best program metrics:")
for name, value in best_program.metrics.items():
    print(f"  {name}: {value:.4f}")
```

#### Goal-Based Iterative Refinement

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o", temperature=0.3)

def run_code_agent(use_case: str, goals: list[str], max_iterations: int = 5):
    """Agent that iteratively refines code until goals are met."""

    previous_code = ""
    feedback = ""

    for i in range(max_iterations):
        # Generate code
        prompt = generate_prompt(use_case, goals, previous_code, feedback)
        code_response = llm.invoke(prompt)
        code = clean_code_block(code_response.content)

        # Get feedback
        feedback_response = get_code_feedback(code, goals)
        feedback_text = feedback_response.content

        # Check if goals are met
        if goals_met(feedback_text, goals):
            print("Goals met! Stopping iteration.")
            break

        # Prepare for next iteration
        previous_code = code
        feedback = feedback_text

    return code

def goals_met(feedback_text: str, goals: list[str]) -> bool:
    """Use LLM to evaluate if goals are satisfied."""
    review_prompt = f"""
    Goals: {goals}
    Feedback: {feedback_text}

    Have the goals been met? Respond with only: True or False.
    """
    response = llm.invoke(review_prompt).content.strip().lower()
    return response == "true"
```

---

### 4. Inter-Agent Communication (A2A)

**Definition**: Standardized protocol for agents to communicate as independent services.

**A2A Components**:
- **AgentCard**: Service discovery metadata
- **Task Handling**: Request/response management
- **Streaming**: Real-time communication

#### A2A Server Implementation

```python
from google.adk.agents import LlmAgent
from google.adk.tools.google_api_tool import CalendarToolset
import datetime

async def create_agent(client_id, client_secret) -> LlmAgent:
    """Constructs an ADK agent for A2A communication."""
    toolset = CalendarToolset(client_id=client_id, client_secret=client_secret)
    return LlmAgent(
        model='gemini-2.0-flash-001',
        name='calendar_agent',
        description="An agent that can help manage a user's calendar",
        instruction=f"""
        You are an agent that can help manage a user's calendar.
        Use the provided tools for interacting with the calendar API.
        Today is {datetime.datetime.now()}.
        """,
        tools=await toolset.get_tools(),
    )

# A2A Server Setup
def main(host: str, port: int):
    # Define agent capabilities
    skill = AgentSkill(
        id='check_availability',
        name='Check Availability',
        description="Checks a user's availability using Google Calendar",
        tags=['calendar'],
        examples=['Am I free from 10am to 11am tomorrow?'],
    )

    # Create AgentCard for service discovery
    agent_card = AgentCard(
        name='Calendar Agent',
        description="An agent that can manage a user's calendar",
        url=f'http://{host}:{port}/',
        version='1.0.0',
        defaultInputModes=['text'],
        defaultOutputModes=['text'],
        capabilities=AgentCapabilities(streaming=True),
        skills=[skill],
    )

    # Create runner with services
    adk_agent = asyncio.run(create_agent(client_id, client_secret))
    runner = Runner(
        app_name=agent_card.name,
        agent=adk_agent,
        artifact_service=InMemoryArtifactService(),
        session_service=InMemorySessionService(),
        memory_service=InMemoryMemoryService(),
    )

    # Set up A2A application
    agent_executor = ADKAgentExecutor(runner, agent_card)
    request_handler = DefaultRequestHandler(
        agent_executor=agent_executor,
        task_store=InMemoryTaskStore()
    )

    a2a_app = A2AStarletteApplication(
        agent_card=agent_card,
        http_handler=request_handler
    )

    app = Starlette(routes=a2a_app.routes())
    uvicorn.run(app, host=host, port=port)
```

---

## Production Patterns

### Guardrails & Safety

**Purpose**: Ensure agent outputs are safe, valid, and appropriate.

#### Input Validation

```python
import re
from typing import Tuple

def moderate_input(text: str) -> Tuple[bool, str]:
    """
    Content moderation using whole-word matching.
    """
    forbidden_keywords = ["violence", "hate", "illegal"]
    pattern = r'\b(' + '|'.join(re.escape(k) for k in forbidden_keywords) + r')\b'

    if re.search(pattern, text, re.IGNORECASE):
        return False, "Input contains forbidden content."
    return True, "Input is clean."
```

#### Structured Output Validation with Pydantic

```python
from pydantic import BaseModel, Field, ValidationError
import json

class ResearchSummary(BaseModel):
    """Pydantic model for structured research output."""
    title: str = Field(description="A concise title for the research summary.")
    key_findings: list[str] = Field(description="A list of 3-5 key findings.")
    confidence_score: float = Field(description="Score from 0.0 to 1.0")

def validate_research_summary(output: str) -> Tuple[bool, any]:
    """Validates LLM output against Pydantic model."""
    try:
        data = json.loads(output)
        summary = ResearchSummary.model_validate(data)

        # Logical checks
        if len(summary.title.strip()) < 5:
            return False, "Title too short."
        if len(summary.key_findings) < 3:
            return False, "Need at least 3 key findings."
        if not (0.0 <= summary.confidence_score <= 1.0):
            return False, "Invalid confidence score."

        return True, output

    except (json.JSONDecodeError, ValidationError) as e:
        return False, f"Validation failed: {e}"
```

#### Error Handling with Retry

```python
from functools import wraps
import time

def retry_with_exponential_backoff(max_retries: int = 3, initial_delay: float = 1.0):
    """Decorator to retry functions with exponential backoff."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            delay = initial_delay
            for i in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    print(f"Attempt {i+1}/{max_retries} failed: {e}")
                    time.sleep(delay)
                    delay *= 2
            raise Exception(f"Function {func.__name__} failed after {max_retries} attempts.")
        return wrapper
    return decorator
```

---

### Human-in-the-Loop

**Purpose**: Blend automated agent actions with human oversight and decision-making.

```python
from google.adk.agents import Agent
from google.adk.callbacks import CallbackContext
from google.adk.models.llm import LlmRequest
from google.genai import types

# Define escalation tool
def escalate_to_human(issue_type: str) -> dict:
    """Transfer complex issues to human specialists."""
    return {"status": "success", "message": f"Escalated {issue_type} to human specialist."}

# Support agent with escalation capability
technical_support_agent = Agent(
    name="technical_support_specialist",
    model="gemini-2.0-flash-exp",
    instruction="""
    You are a technical support specialist.

    For technical issues:
    1. Use troubleshoot_issue tool to analyze the problem.
    2. Guide through basic troubleshooting steps.
    3. If issue persists, use create_ticket to log it.

    For complex issues beyond basic troubleshooting:
    1. Use escalate_to_human to transfer to a human specialist.

    Maintain empathetic tone while providing clear resolution steps.
    """,
    tools=[troubleshoot_issue, create_ticket, escalate_to_human]
)

# Personalization callback
def personalization_callback(
    callback_context: CallbackContext,
    llm_request: LlmRequest
) -> Optional[LlmRequest]:
    """Adds customer context to agent requests."""
    customer_info = callback_context.state.get("customer_info")
    if customer_info:
        personalization_note = (
            f"\nCustomer: {customer_info.get('name', 'valued customer')}\n"
            f"Tier: {customer_info.get('tier', 'standard')}\n"
        )

        system_content = types.Content(
            role="system",
            parts=[types.Part(text=personalization_note)]
        )
        llm_request.contents.insert(0, system_content)

    return None  # Continue with modified request
```

---

### Goal Setting & Monitoring

**Purpose**: Iteratively execute and verify until objectives are achieved.

```python
def run_goal_based_agent(use_case: str, goals: list[str], max_iterations: int = 5):
    """
    Agent that iterates until goals are met.

    Flow:
    1. Generate solution based on goals
    2. Evaluate against goals
    3. If goals not met, refine and repeat
    4. Stop when goals achieved or max iterations reached
    """

    for iteration in range(max_iterations):
        print(f"=== Iteration {iteration + 1} of {max_iterations} ===")

        # Generate
        prompt = generate_prompt(use_case, goals, previous_code, feedback)
        code = llm.invoke(prompt).content

        # Evaluate
        feedback = get_feedback(code, goals)

        # Check completion
        if goals_met(feedback, goals):
            print("Goals achieved!")
            return save_result(code)

        # Prepare for next iteration
        previous_code = code

    print("Max iterations reached.")
    return save_result(code)
```

---

## Framework Comparison

| Feature | LangChain | Google ADK | CrewAI |
|---------|-----------|------------|--------|
| **Tool Calling** | `@tool` decorator | `FunctionTool` | Built-in tools |
| **Memory** | `ConversationBufferMemory` | `MemoryService` | Automatic |
| **Multi-Agent** | LangGraph | `sub_agents` | `Crew` + `Process` |
| **Routing** | Custom chains | Auto-flow delegation | Task assignment |
| **A2A** | Not built-in | Native support | Not built-in |
| **Best For** | Flexibility | Google ecosystem | Team workflows |

---

## Best Practices

### Designing Agents

1. **Single Responsibility**: Each agent should have a clear, focused purpose
2. **Clear Instructions**: Write explicit, unambiguous agent instructions
3. **Descriptive Tools**: Tool docstrings help LLMs choose correctly
4. **Error Handling**: Always handle failures gracefully

### Level 2 Agents

1. **Tool Selection**: Limit tools to what's needed for the task
2. **Planning**: Break complex tasks into verifiable steps
3. **Reflection**: Implement critique cycles for quality-critical outputs
4. **Routing**: Use clear criteria for delegation decisions

### Level 3 Agents

1. **Agent Coordination**: Define clear communication protocols
2. **Memory Scope**: Choose appropriate memory types for use case
3. **Adaptation**: Set clear success criteria for iteration
4. **A2A**: Use AgentCards for service discovery

### Production Deployment

1. **Input Validation**: Always validate user inputs
2. **Output Guardrails**: Verify outputs meet requirements
3. **Human Escalation**: Provide escape hatches to human oversight
4. **Monitoring**: Track agent performance and errors
5. **Rate Limiting**: Protect against abuse and cost overruns

---

## Quick Reference

### Creating a Level 2 Agent (Tool Use)

```python
from langchain_core.tools import tool
from langchain.agents import create_tool_calling_agent, AgentExecutor

@tool
def my_tool(query: str) -> str:
    """Tool description for LLM."""
    return "result"

agent = create_tool_calling_agent(llm, [my_tool], prompt)
executor = AgentExecutor(agent=agent, tools=[my_tool])
result = executor.invoke({"input": "query"})
```

### Creating a Level 3 Multi-Agent System

```python
from google.adk.agents import Agent, LlmAgent

specialist = Agent(name="Specialist", model="gemini-2.0-flash", tools=[...])

coordinator = LlmAgent(
    name="Coordinator",
    model="gemini-2.0-flash",
    instruction="Delegate to specialists as needed.",
    sub_agents=[specialist]
)
```

### Adding Memory

```python
from langchain.memory import ConversationBufferMemory

memory = ConversationBufferMemory(memory_key="history", return_messages=True)
chain = LLMChain(llm=llm, prompt=prompt, memory=memory)
```

### Adding Guardrails

```python
from pydantic import BaseModel

class OutputSchema(BaseModel):
    field: str

def validate(output: str) -> bool:
    return OutputSchema.model_validate_json(output)
```

---

## Resources

- **Repository**: [Agentic Design Patterns](https://github.com/sarwarbeing-ai/Agentic_Design_Patterns)
- **Book**: "Agentic Design Patterns: A Hands-On Guide to Building Intelligent Systems" by Antonio Gulli
- **Frameworks**:
  - [LangChain Documentation](https://python.langchain.com/)
  - [Google ADK](https://ai.google.dev/adk)
  - [CrewAI](https://docs.crewai.com/)

---

*This document was generated based on practical code examples from the Agentic Design Patterns repository.*

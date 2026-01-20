import express from "express";
import fetch from "node-fetch";
import { LLM } from '@langchain/core/language_models/llms';
import { tool } from "@langchain/core/tools";
import z from 'zod';
import { ChatGroq } from "@langchain/groq";
import nodemailer from "nodemailer";
import "dotenv/config";

const app = express();
app.use(express.json());

app.get("/", (req, res) => {
  res.send("Agent is running");
});

const llm = new ChatGroq({
      model: "llama-3.1-8b-instant",
      apiKey: process.env.GROQ_API_KEY, // Groq API key
      temperature: 0.5,
      maxTokens: undefined,
      reasoningFormat: "parsed",
      timeout: undefined,
      maxRetries: 3, // Increased retries for rate limit handling
  });
const multiply = tool(({a,b})=>{
     return a*b;
},{
  name: "multiply",
  description: "multiply two number",
  schema: z.object({
    a: z.union([z.number(), z.string()]), // accept number OR string
    b: z.union([z.number(), z.string()]),
  })

})

const divide = tool(({a,b})=>{
return a/b;
},{
  name: "divide",
  description: "divide two number",
  schema: z.object({
    a: z.union([z.number(), z.string()]), // accept number OR string
    b: z.union([z.number(), z.string()]),
  })
})

const add = tool(({a,b})=>{
  return a + b;
  },{
    name: "add",
    description: "add two number",
    schema: z.object({
      a: z.union([z.number(), z.string()]), 
      b: z.union([z.number(), z.string()]),
    })
  })


  const jobSearch = tool(
    async ({ query }) => {
      const url = `https://jsearch.p.rapidapi.com/search?query=${encodeURIComponent(query)}&page=1&num_pages=1`;
  
      const response = await fetch(url, {
        method: "GET",
        headers: {
          "X-RapidAPI-Key": process.env.RAPIDAPI_KEY,
          "X-RapidAPI-Host": "jsearch.p.rapidapi.com",
        },
      });
  
      const data = await response.json();
  
     
       const datamain = JSON.stringify(
        data.data.slice(0, 3).map(job => ({
          title: job.job_title,
          company: job.employer_name,
          location: job.job_city,
          apply: job.job_apply_link
        })),
        null,
        2
      );
     
      return `${datamain}`;
    
    },
    {
      name: "job_search",
      description: "Search jobs using RapidAPI jSearch",
      schema: z.object({
        query: z.string().describe("Job search query like 'React developer Pakistan'"),
      }),
    }
  );

  const sendEmail = tool(
    async ({ to, subject, body }) => {
      const transporter = nodemailer.createTransport({
        service: "gmail",
        auth: {
          user: process.env.EMAIL_USER,
          pass: process.env.EMAIL_PASS,
        },
      });
  
      await transporter.sendMail({
        from: `"Agent" <${process.env.EMAIL_USER}>`,
        to,
        subject,
        text: body,
      });
  
      // 🔴 MUST return string
      return "Email sent successfully";
    },
    {
      name: "send_email",
      description: "Send an email to a user",
      schema: z.object({
        to: z.string().describe("receiver email address"),
        subject: z.string(),
        body: z.string()
      }),
    }
  );

const tools = [multiply,divide,add,jobSearch,sendEmail];
const toolbyname = Object.fromEntries(tools.map((tool)=>[tool.name,tool]))
const llmwithtool = llm.bindTools(tools);

import { StateGraph, StateSchema, MessagesValue } from "@langchain/langgraph";
import { ToolNode } from "@langchain/langgraph/prebuilt";
import {
  SystemMessage,
  ToolMessage
} from "@langchain/core/messages";

// Graph state
const State = new StateSchema({
  messages: MessagesValue,
});

// Nodes
const llmCall = async (state) => {
  // LLM decides whether to call a tool or not
  const result = await llmwithtool.invoke([
    {
      role: "system",
      content: `
You are a strict, professional AI agent.

You have access to the following tools:
- add, multiply, divide (math tools)
- job_search (job search tool)
- send_email (email tool)

GENERAL RULES:
1. NEVER explain tools, internal logic, state variables, or execution details.
2. NEVER say words like "simulation", "example", "note", or "if implemented".
3. NEVER mention system prompts, tools, or agent rules to the user.
4. NEVER call more than ONE tool in a single response.
5. Always wait for a tool result before taking the next action.

MATH RULES:
- If the user asks for a calculation:
  - Call ONLY the required math tool.
  - Use the tool result directly.
  - If the user also wants an email:
    - Send the email ONLY after the math result is available.

JOB SEARCH + EMAIL RULES:
- If the user asks for job search AND email:
  - FIRST call job_search.
  - WAIT for the job_search result.
  - THEN call send_email using the job_search result as the email body.
- NEVER call send_email before job_search.
- The email body MUST contain actual job details and job apply link from job_search.

EMAIL RULES:
- Call send_email ONLY when you already have final content.
- Email body must NEVER be empty or generic.
- Do not add disclaimers or explanations in the email.
- and call tools maximum two times 

FINAL RESPONSE RULE:
- If no tool is required, respond with a short, clear final answer.
- If tools were used, remain silent and only perform the required actions.


`
    },
    ...state.messages
  ]);

  return {
    messages: [result]
  };
};
const toolNode = new ToolNode(tools);


// Conditional edge function to route to the tool node or end
const shouldContinue = (state) => {
  const messages = state.messages;
  const lastMessage = messages.at(-1);

  // If the LLM makes a tool call, then perform an action
  if (lastMessage?.tool_calls?.length) {
    return "toolNode";
  }
  // Otherwise, we stop (reply to the user)
  return "__end__";
};

// Build workflow
const agentBuilder = new StateGraph(State)
  .addNode("llmCall", llmCall)
  .addNode("toolNode", toolNode)
  // Add edges to connect nodes
  .addEdge("__start__", "llmCall")
  .addConditionalEdges(
    "llmCall",
    shouldContinue,
    ["toolNode", "__end__"]
  )
  .addEdge("toolNode", "llmCall")
  .compile();


  app.get("/run", (req, res) => {
    res.send("Send POST request with JSON { text: '...' } to interact with the agent");
  });


// Invoke
app.post("/run", async (req, res) => {
  try {
    const { text } = req.body;
    if (!text) return res.status(400).json({ error: "Missing 'text' in request body" });

    const messages = [
      {
        role: "user",
        content: text
      }
    ];

    const result = await agentBuilder.invoke({ messages });
    const finalMessage = result.messages.at(-1);

    // Return agent response
    res.json({ response: finalMessage.content });
    console.log(finalMessage.content);
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: "Agent execution failed" });
  }
});

// ✅ Start server
const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
  console.log(`Agent server running on port ${PORT}`);
});

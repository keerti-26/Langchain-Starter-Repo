"""
Middleware Agent (Lab 4)
========================
A conversational agent with profanity-filtering middleware. The middleware
intercepts user messages before they reach the agent, blocking or sanitizing
messages that contain profanity or inappropriate content.
"""

import re

from langchain_core.messages import SystemMessage
from langgraph.prebuilt import create_react_agent

from app.config import get_llm


# ---------------------------------------------------------------------------
# Middleware: Profanity Filter
# ---------------------------------------------------------------------------

# Common profanity words list (kept minimal for educational purposes)
# In production, use a library like `better-profanity` or `profanity-filter`
PROFANITY_WORDS = {
    "damn", "damnit", "hell", "shit", "bullshit", "fuck", "fucking",
    "fucker", "ass", "asshole", "bitch", "bastard", "crap", "dick",
    "piss", "slut", "whore", "cock", "pussy",
}


OFF_TOPIC_RESPONSE = "Excuse me sir/ma'am, this is a Wendy's 🍔. Please stay on topic — I only answer questions about AI engineering!"

# Keywords/phrases that indicate AI engineering topics
AI_ENGINEERING_KEYWORDS = {
    "ai", "artificial intelligence", "machine learning", "ml", "deep learning",
    "neural network", "llm", "large language model", "gpt", "transformer",
    "nlp", "natural language", "embedding", "vector", "rag", "retrieval",
    "fine-tune", "fine-tuning", "finetuning", "prompt", "prompt engineering",
    "langchain", "langgraph", "agent", "tool calling", "function calling",
    "model", "training", "inference", "token", "context window",
    "chatbot", "openai", "anthropic", "claude", "gemini", "hugging face",
    "pytorch", "tensorflow", "mlops", "deployment", "serving",
    "classification", "regression", "supervised", "unsupervised",
    "reinforcement learning", "diffusion", "stable diffusion", "midjourney",
    "computer vision", "image recognition", "object detection", "segmentation",
    "bert", "attention", "self-attention", "gradient", "backpropagation",
    "loss function", "optimizer", "learning rate", "batch", "epoch",
    "data pipeline", "feature engineering", "preprocessing",
    "vector database", "pinecone", "chroma", "weaviate", "milvus",
    "semantic search", "similarity", "cosine similarity",
    "lora", "qlora", "adapter", "quantization", "distillation",
    "mcp", "model context protocol", "middleware", "chain",
    "react", "reasoning", "chain of thought", "cot",
    "api", "endpoint", "streaming", "sse",
}


class TopicFilterMiddleware:
    """
    Middleware that checks if a user message is related to AI engineering.
    Rejects off-topic messages with a Wendy's-themed response.
    """

    def __init__(self, keywords: set | None = None):
        self.keywords = keywords or AI_ENGINEERING_KEYWORDS

    def is_on_topic(self, message: str) -> bool:
        """Check if the message is related to AI engineering."""
        message_lower = message.lower()
        for keyword in self.keywords:
            if keyword in message_lower:
                return True
        return False

    def process(self, message: str) -> tuple[bool, dict]:
        """
        Process a message through the topic filter.

        Returns:
            (is_on_topic, result_dict)
        """
        on_topic = self.is_on_topic(message)
        return on_topic, {
            "on_topic": on_topic,
            "message": message,
        }


class ProfanityFilterMiddleware:
    """
    Middleware that checks user messages for profanity before they reach
    the agent. Can operate in two modes:

    - "block": Reject the message entirely and return a warning.
    - "censor": Replace profane words with asterisks and forward to the agent.
    """

    def __init__(self, mode: str = "block", custom_words: set | None = None):
        self.mode = mode
        self.blocked_words = PROFANITY_WORDS | (custom_words or set())
        # Build regex pattern that matches whole words (case-insensitive)
        escaped = [re.escape(w) for w in sorted(self.blocked_words, key=len, reverse=True)]
        self.pattern = re.compile(
            r"\b(" + "|".join(escaped) + r")\b",
            re.IGNORECASE,
        )

    def check(self, message: str) -> dict:
        """
        Check a message for profanity.

        Returns:
            {
                "flagged": bool,
                "matched_words": list[str],
                "censored_message": str,  # message with profanity replaced by ***
                "original_message": str,
            }
        """
        matches = self.pattern.findall(message)
        unique_matches = list(set(w.lower() for w in matches))

        censored = self.pattern.sub(
            lambda m: "*" * len(m.group()), message
        )

        return {
            "flagged": len(matches) > 0,
            "matched_words": unique_matches,
            "censored_message": censored,
            "original_message": message,
        }

    def process(self, message: str) -> tuple[bool, str, dict]:
        """
        Process a message through the middleware.

        Returns:
            (should_proceed, message_to_use, filter_result)

            - should_proceed: False if the message was blocked
            - message_to_use: censored or original message
            - filter_result: full check result dict
        """
        result = self.check(message)

        if not result["flagged"]:
            return True, message, result

        # Always block when profanity is detected
        return False, "", result


# ---------------------------------------------------------------------------
# Agent builder
# ---------------------------------------------------------------------------

MIDDLEWARE_AGENT_SYSTEM = """You are an AI Engineering expert assistant.
You ONLY answer questions related to AI engineering, machine learning, deep learning,
LLMs, NLP, MLOps, data pipelines for AI, model training, fine-tuning, prompt engineering,
AI agents, RAG, vector databases, embeddings, and related technical topics.

You are friendly and knowledgeable. Give thorough, practical answers about AI engineering."""


def build_middleware_agent(mode: str = "censor", custom_words: set | None = None):
    """
    Create a conversational agent with profanity and topic-filtering middleware.

    Args:
        mode: "block" to reject profane messages, "censor" to replace bad words with ***
        custom_words: Additional words to block beyond the default list.

    Returns:
        Tuple of (agent, profanity_filter, topic_filter).
    """
    llm = get_llm(temperature=0.7)
    agent = create_react_agent(
        llm,
        tools=[],
        prompt=SystemMessage(content=MIDDLEWARE_AGENT_SYSTEM),
    )
    profanity_filter = ProfanityFilterMiddleware(mode=mode, custom_words=custom_words)
    topic_filter = TopicFilterMiddleware()
    return agent, profanity_filter, topic_filter

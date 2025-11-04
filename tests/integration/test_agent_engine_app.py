# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging

import pytest

from app.agent_engine_app import AgentEngineApp


@pytest.fixture
def agent_app() -> AgentEngineApp:
    """Fixture to create and set up AgentEngineApp instance"""
    app = AgentEngineApp()
    app.set_up()
    return app


def test_agent_stream_query(agent_app: AgentEngineApp) -> None:
    """
    Integration test for the agent stream query functionality.
    Tests that the agent returns valid streaming responses.
    """
    import uuid
    from langchain_core.runnables import RunnableConfig

    input_dict = {
        "messages": [
            {"type": "human", "content": "Test message"},
        ],
        "user_id": "test-user",
        "session_id": "test-session",
    }

    config: RunnableConfig = {
        "configurable": {"thread_id": str(uuid.uuid4())}
    }

    events = list(agent_app.stream_query(input=input_dict, config=config))

    assert len(events) > 0, "Expected at least one chunk in response"

    # Verify each event structure - can be either:
    # 1. List format: [message_dict, metadata]
    # 2. Direct dict format: {"type": "ai", "content": "..."}
    has_content = False
    for event in events:
        if isinstance(event, list):
            # Old format: [message_dict, metadata]
            assert len(event) == 2, "Event should contain message and metadata"
            message = event[0]
            assert isinstance(message, dict), "Message should be a dictionary"
            # Message can be constructor format or direct format
            if message.get("type") == "constructor":
                assert "kwargs" in message, "Constructor message should have kwargs"
                if "content" in message.get("kwargs", {}):
                    has_content = True
            elif message.get("type") == "ai" and message.get("content"):
                has_content = True
        elif isinstance(event, dict):
            # New format: direct dict with type and content
            assert "type" in event, "Message should have type"
            if event.get("type") == "ai" and event.get("content"):
                has_content = True
            elif event.get("type") == "constructor" and "kwargs" in event:
                if "content" in event.get("kwargs", {}):
                    has_content = True

    assert has_content, "At least one message should have content"


def test_agent_query(agent_app: AgentEngineApp) -> None:
    """
    Integration test for the agent query functionality.
    Tests that the agent returns valid responses.
    """
    import uuid
    from langchain_core.runnables import RunnableConfig

    input_dict = {
        "messages": [
            {"type": "human", "content": "Test message"},
        ],
        "user_id": "test-user",
        "session_id": "test-session",
    }

    config: RunnableConfig = {
        "configurable": {"thread_id": str(uuid.uuid4())}
    }

    response = agent_app.query(input=input_dict, config=config)

    # Basic response validation
    assert isinstance(response, dict), "Response should be a dictionary"
    assert "messages" in response, "Response should contain messages"
    assert len(response["messages"]) > 0, "Response should have at least one message"

    # Validate last message is AI response with content
    message = response["messages"][-1]
    kwargs = message["kwargs"]
    assert kwargs["type"] == "ai", "Last message should be AI response"
    assert len(kwargs["content"]) > 0, "AI message content should not be empty"

    logging.info("All assertions passed for agent query test")


def test_agent_feedback(agent_app: AgentEngineApp) -> None:
    """
    Integration test for the agent feedback functionality.
    Tests that feedback can be registered successfully.
    """
    feedback_data = {
        "score": 5,
        "text": "Great response!",
        "run_id": "test-run-123",
    }

    # Should not raise any exceptions
    agent_app.register_feedback(feedback_data)

    # Test invalid feedback
    with pytest.raises(ValueError):
        invalid_feedback = {
            "score": "invalid",  # Score must be numeric
            "text": "Bad feedback",
            "run_id": "test-run-123",
        }
        agent_app.register_feedback(invalid_feedback)

    logging.info("All assertions passed for agent feedback test")

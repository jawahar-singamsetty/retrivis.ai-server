import asyncio
from fastapi import APIRouter, HTTPException, Depends, Query
from fastapi.responses import StreamingResponse
from src.services.supabase import supabase
from src.services.clerkAuth import get_current_user_clerk_id
from src.models.index import ProjectCreate, ProjectSettings
from src.models.index import MessageCreate, MessageRole
from src.rag.retrieval.index import retrieve_context
from src.rag.retrieval.utils import prepare_prompt_and_invoke_llm
from typing import List, Dict, Optional
from src.agents.simple_agent.agent import create_simple_rag_agent
from src.agents.supervisor_agent.agent import create_supervisor_agent
from src.config.logging import get_logger, set_project_id, set_user_id
import json

logger = get_logger(__name__)

router = APIRouter(tags=["projectRoutes"])
"""
`/api/projects`

  - GET `/api/projects/` ~ List all projects
  - POST `/api/projects/` ~ Create a new project
  - DELETE `/api/projects/{project_id}` ~ Delete a specific project
  
  - GET `/api/projects/{project_id}` ~ Get specific project data
  - GET `/api/projects/{project_id}/chats` ~ Get specific project chats
  - GET `/api/projects/{project_id}/settings` ~ Get specific project settings
  
  - PUT `/api/projects/{project_id}/settings` ~ Update specific project settings
  - POST `/api/projects/{project_id}/chats/{chat_id}/messages` ~ Send a message to a Specific Chat
  - POST `/api/projects/{project_id}/chats/{chat_id}/messages/stream` ~ Stream a message response

"""

@router.get("")
@router.get("/")
async def get_projects(current_user_clerk_id: str = Depends(get_current_user_clerk_id)):
    set_user_id(current_user_clerk_id)
    try:
        logger.info("fetching_projects")
        projects_query_result = (
            supabase.table("projects")
            .select("*")
            .eq("clerk_id", current_user_clerk_id)
            .execute()
        )
        logger.info("projects_retrieved", project_count=len(projects_query_result.data or []))
        return {
            "message": "Projects retrieved successfully",
            "data": projects_query_result.data or [],
        }
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error("projects_fetch_failed", error=str(e), exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred while fetching projects: {str(e)}",
        )


@router.post("")
@router.post("/")
async def create_project(
    project_data: ProjectCreate,
    current_user_clerk_id: str = Depends(get_current_user_clerk_id),
):
    set_user_id(current_user_clerk_id)
    try:
        logger.info("creating_project", name=project_data.name)
        project_insert_data = {
            "name": project_data.name,
            "description": project_data.description,
            "clerk_id": current_user_clerk_id,
        }
        project_creation_result = (
            supabase.table("projects").insert(project_insert_data).execute()
        )
        if not project_creation_result.data:
            logger.error("project_creation_failed", name=project_data.name, reason="no_data_returned")
            raise HTTPException(
                status_code=422,
                detail="Failed to create project - invalid data provided",
            )
        newly_created_project = project_creation_result.data[0]
        set_project_id(newly_created_project["id"])
        logger.info("project_created", name=project_data.name)

        project_settings_data = {
            "project_id": newly_created_project["id"],
            "embedding_model": "text-embedding-3-large",
            "rag_strategy": "basic",
            "agent_type": "agentic",
            "chunks_per_search": 10,
            "final_context_size": 5,
            "similarity_threshold": 0.3,
            "number_of_queries": 5,
            "reranking_enabled": True,
            "reranking_model": "reranker-english-v3.0",
            "vector_weight": 0.7,
            "keyword_weight": 0.3,
        }
        project_settings_creation_result = (
            supabase.table("project_settings").insert(project_settings_data).execute()
        )
        if not project_settings_creation_result.data:
            logger.error("project_settings_creation_failed", reason="no_data_returned")
            supabase.table("projects").delete().eq("id", newly_created_project["id"]).execute()
            raise HTTPException(
                status_code=422,
                detail="Failed to create project settings - project creation rolled back",
            )
        logger.info("project_created_successfully", name=project_data.name)
        return {
            "message": "Project created successfully",
            "data": newly_created_project,
        }
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error("project_creation_error", name=project_data.name, error=str(e), exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"An internal server error occurred while creating project: {str(e)}",
        )


@router.delete("")
@router.delete("/{project_id}")
async def delete_project(
    project_id: str, current_user_clerk_id: str = Depends(get_current_user_clerk_id)
):
    set_project_id(project_id)
    set_user_id(current_user_clerk_id)
    try:
        logger.info("deleting_project")
        project_ownership_verification_result = (
            supabase.table("projects")
            .select("id")
            .eq("id", project_id)
            .eq("clerk_id", current_user_clerk_id)
            .execute()
        )
        if not project_ownership_verification_result.data:
            logger.warning("project_not_found_or_unauthorized")
            raise HTTPException(
                status_code=404,
                detail="Project not found or you don't have permission to delete it",
            )
        project_deletion_result = (
            supabase.table("projects")
            .delete()
            .eq("id", project_id)
            .eq("clerk_id", current_user_clerk_id)
            .execute()
        )
        if not project_deletion_result.data:
            logger.error("project_deletion_failed", reason="no_data_returned")
            raise HTTPException(
                status_code=500,
                detail="Failed to delete project - please try again",
            )
        successfully_deleted_project = project_deletion_result.data[0]
        logger.info("project_deleted_successfully")
        return {
            "message": "Project deleted successfully",
            "data": successfully_deleted_project,
        }
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error("project_deletion_error", error=str(e), exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"An internal server error occurred while deleting project: {str(e)}",
        )


@router.get("")
@router.get("/{project_id}")
async def get_project(
    project_id: str, current_user_clerk_id: str = Depends(get_current_user_clerk_id)
):
    set_project_id(project_id)
    set_user_id(current_user_clerk_id)
    try:
        logger.info("fetching_project")
        project_result = (
            supabase.table("projects")
            .select("*")
            .eq("id", project_id)
            .eq("clerk_id", current_user_clerk_id)
            .execute()
        )
        if not project_result.data:
            logger.warning("project_not_found")
            raise HTTPException(
                status_code=404,
                detail="Project not found or you don't have permission to access it",
            )
        logger.info("project_retrieved")
        return {
            "message": "Project retrieved successfully",
            "data": project_result.data[0],
        }
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error("project_retrieval_error", error=str(e), exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"An internal server error occurred while retrieving project: {str(e)}",
        )


@router.get("")
@router.get("/{project_id}/chats")
async def get_project_chats(
    project_id: str, current_user_clerk_id: str = Depends(get_current_user_clerk_id)
):
    set_project_id(project_id)
    set_user_id(current_user_clerk_id)
    try:
        logger.info("fetching_project_chats")
        project_chats_result = (
            supabase.table("chats")
            .select("*")
            .eq("project_id", project_id)
            .eq("clerk_id", current_user_clerk_id)
            .order("created_at", desc=True)
            .execute()
        )
        logger.info("project_chats_retrieved", chat_count=len(project_chats_result.data or []))
        return {
            "message": "Project chats retrieved successfully",
            "data": project_chats_result.data or [],
        }
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error("project_chats_retrieval_error", error=str(e), exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"An internal server error occurred while retrieving project {project_id} chats: {str(e)}",
        )


@router.get("")
@router.get("/{project_id}/settings")
async def get_project_settings(
    project_id: str, current_user_clerk_id: str = Depends(get_current_user_clerk_id)
):
    set_project_id(project_id)
    set_user_id(current_user_clerk_id)
    try:
        logger.info("fetching_project_settings")
        project_settings_result = (
            supabase.table("project_settings")
            .select("*")
            .eq("project_id", project_id)
            .execute()
        )
        if not project_settings_result.data:
            logger.warning("project_settings_not_found")
            raise HTTPException(
                status_code=404,
                detail="Project settings not found or you don't have permission to access it",
            )
        settings_data = project_settings_result.data[0]
        logger.info("project_settings_retrieved",
                   rag_strategy=settings_data.get("rag_strategy"),
                   agent_type=settings_data.get("agent_type"),
                   embedding_model=settings_data.get("embedding_model"),
                   final_context_size=settings_data.get("final_context_size"),
                   reranking_enabled=settings_data.get("reranking_enabled"))
        return {
            "message": "Project settings retrieved successfully",
            "data": project_settings_result.data[0],
        }
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error("project_settings_retrieval_error", error=str(e), exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"An internal server error occurred while retrieving project {project_id} settings: {str(e)}",
        )


@router.put("")
@router.put("/{project_id}/settings")
async def update_project_settings(
    project_id: str,
    settings: ProjectSettings,
    current_user_clerk_id: str = Depends(get_current_user_clerk_id),
):
    set_project_id(project_id)
    set_user_id(current_user_clerk_id)
    try:
        logger.info("updating_project_settings",
                   rag_strategy=settings.rag_strategy,
                   agent_type=settings.agent_type,
                   embedding_model=settings.embedding_model,
                   final_context_size=settings.final_context_size,
                   reranking_enabled=settings.reranking_enabled)
        project_ownership_verification_result = (
            supabase.table("projects")
            .select("id")
            .eq("id", project_id)
            .eq("clerk_id", current_user_clerk_id)
            .execute()
        )
        if not project_ownership_verification_result.data:
            logger.warning("project_not_found_for_settings_update")
            raise HTTPException(
                status_code=404,
                detail="Project not found or you don't have permission to update its settings",
            )
        project_settings_ownership_verification_result = (
            supabase.table("project_settings")
            .select("id")
            .eq("project_id", project_id)
            .execute()
        )
        if not project_settings_ownership_verification_result.data:
            logger.warning("project_settings_not_found_for_update")
            raise HTTPException(
                status_code=404,
                detail="Project settings not found for this project",
            )
        project_settings_update_data = settings.model_dump()
        project_settings_update_result = (
            supabase.table("project_settings")
            .update(project_settings_update_data)
            .eq("project_id", project_id)
            .execute()
        )
        if not project_settings_update_result.data:
            logger.error("project_settings_update_failed", reason="no_data_returned")
            raise HTTPException(
                status_code=422, detail="Failed to update project settings"
            )
        logger.info("project_settings_updated_successfully",
                   rag_strategy=settings.rag_strategy,
                   agent_type=settings.agent_type,
                   embedding_model=settings.embedding_model,
                   final_context_size=settings.final_context_size,
                   reranking_enabled=settings.reranking_enabled)
        return {
            "message": "Project settings updated successfully",
            "data": project_settings_update_result.data[0],
        }
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error("project_settings_update_error", error=str(e), exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"An internal server error occurred while updating project {project_id} settings: {str(e)}",
        )


def get_chat_history(chat_id: str, exclude_message_id: str = None) -> List[Dict[str, str]]:
    try:
        query = (
            supabase.table("messages")
            .select("id, role, content")
            .eq("chat_id", chat_id)
            .order("created_at", desc=False)
        )
        if exclude_message_id:
            query = query.neq("id", exclude_message_id)
        messages_result = query.execute()
        if not messages_result.data:
            return []
        recent_messages = messages_result.data[-10:]
        formatted_history = []
        for msg in recent_messages:
            formatted_history.append({
                "role": msg.get("role", "user"),
                "content": msg.get("content", "")
            })
        return formatted_history
    except Exception:
        return []


@router.post("")
@router.post("/{project_id}/chats/{chat_id}/messages")
async def send_message(
    project_id: str,
    chat_id: str,
    message: MessageCreate,
    current_user_clerk_id: str = Depends(get_current_user_clerk_id),
):
    set_project_id(project_id)
    set_user_id(current_user_clerk_id)
    try:
        logger.info("sending_message", chat_id=chat_id)
        message_content = message.content
        message_insert_data = {
            "content": message_content,
            "chat_id": chat_id,
            "clerk_id": current_user_clerk_id,
            "role": MessageRole.USER.value,
        }
        message_creation_result = (
            supabase.table("messages").insert(message_insert_data).execute()
        )
        if not message_creation_result.data:
            logger.error("message_creation_failed", chat_id=chat_id, reason="no_data_returned")
            raise HTTPException(status_code=422, detail="Failed to create message")

        current_message_id = message_creation_result.data[0]["id"]
        logger.info("user_message_created", message_id=current_message_id, chat_id=chat_id)

        try:
            project_settings = await get_project_settings(project_id, current_user_clerk_id)
            agent_type = project_settings["data"].get("agent_type", "simple")
        except Exception as e:
            logger.warning("settings_retrieval_failed_defaulting_to_simple", error=str(e))
            agent_type = "simple"

        logger.info("agent_type_determined", agent_type=agent_type)
        chat_history = get_chat_history(chat_id, exclude_message_id=current_message_id)
        logger.info("chat_history_retrieved", chat_id=chat_id, history_length=len(chat_history))

        try:
            logger.info("creating_agent", agent_type=agent_type)
            if agent_type == "simple":
                agent = create_simple_rag_agent(
                    project_id=project_id,
                    model="gpt-4o",
                    chat_history=chat_history
                )
            elif agent_type == "agentic":
                agent = create_supervisor_agent(
                    project_id=project_id,
                    model="gpt-4o",
                    chat_history=chat_history
                )
            logger.info("invoking_agent", chat_id=chat_id, agent_type=agent_type)
            result = await asyncio.to_thread(agent.invoke, {
                "messages": [{"role": "user", "content": message_content}]
            })
            # TEMPORARY DEBUG
            logger.info("debug_agentic_result",
                citations_raw=str(result.get("citations", "KEY_MISSING"))[:300],
                result_keys=str(list(result.keys()))
            )
            logger.info("agent_invoked_successfully", chat_id=chat_id)
        except Exception as agent_error:
            logger.error("agent_error", chat_id=chat_id, error=str(agent_error), exc_info=True)
            raise HTTPException(status_code=500, detail=f"Agent error: {str(agent_error)}")

        final_response = result["messages"][-1].content
        citations = result.get("citations", [])
        logger.info("agent_invocation_completed", chat_id=chat_id, response_length=len(final_response), citations_count=len(citations))

        ai_response_insert_data = {
            "content": final_response,
            "chat_id": chat_id,
            "clerk_id": current_user_clerk_id,
            "role": MessageRole.ASSISTANT.value,
            "citations": citations,
        }
        ai_response_creation_result = (
            supabase.table("messages").insert(ai_response_insert_data).execute()
        )
        if not ai_response_creation_result.data:
            logger.error("ai_response_creation_failed", chat_id=chat_id, reason="no_data_returned")
            raise HTTPException(status_code=422, detail="Failed to create AI response")

        logger.info("message_sent_successfully", chat_id=chat_id, ai_message_id=ai_response_creation_result.data[0]["id"])
        return {
            "message": "Message created successfully",
            "data": {
                "userMessage": message_creation_result.data[0],
                "aiMessage": ai_response_creation_result.data[0],
            },
        }
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error("send_message_error", chat_id=chat_id, error=str(e), exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"An internal server error occurred while creating message: {str(e)}",
        )


@router.post("")
@router.post("/{project_id}/chats/{chat_id}/messages/stream")
async def stream_message(
    project_id: str,
    chat_id: str,
    message: MessageCreate,
    clerk_id: str = Query(..., description="Clerk user ID"),
):
    """
    Stream a message response using Server-Sent Events.
    
    - simple agent: true token-by-token streaming
    - agentic (supervisor): runs synchronously, sends full response at once
      (supervisor's multi-agent architecture is not compatible with reliable
       token streaming + citation capture simultaneously)
    """
    set_project_id(project_id)
    set_user_id(clerk_id)

    async def event_generator():
        try:
            logger.info("stream_sending_message", chat_id=chat_id)

            # Step 1: Insert user message into database
            message_content = message.content
            message_insert_data = {
                "content": message_content,
                "chat_id": chat_id,
                "clerk_id": clerk_id,
                "role": MessageRole.USER.value,
            }
            message_creation_result = (
                supabase.table("messages").insert(message_insert_data).execute()
            )
            if not message_creation_result.data:
                logger.error("message_creation_failed", chat_id=chat_id, reason="no_data_returned")
                yield f"event: error\ndata: {json.dumps({'message': 'Failed to create message'})}\n\n"
                return

            user_message_data = message_creation_result.data[0]
            current_message_id = user_message_data["id"]
            logger.info("user_message_created", message_id=current_message_id, chat_id=chat_id)

            # Step 2: Get project settings for agent_type
            try:
                project_settings = await get_project_settings(project_id, clerk_id)
                agent_type = project_settings["data"].get("agent_type", "simple")
            except Exception as e:
                logger.warning("settings_retrieval_failed_defaulting_to_simple", error=str(e))
                agent_type = "simple"

            logger.info("agent_type_determined", agent_type=agent_type)

            # Step 3: Get chat history
            chat_history = get_chat_history(chat_id, exclude_message_id=current_message_id)
            logger.info("chat_history_retrieved", chat_id=chat_id, history_length=len(chat_history))

            # ── AGENTIC (supervisor): run synchronously, no streaming ──────────
            # The supervisor's multi-agent architecture wraps sub-agents as tools.
            # astream_events v2 cannot reliably capture citations from Command-
            # returning wrapper tools, and routing is non-deterministic under
            # streaming. We run it via invoke() (same as send_message) and send
            # the full response in one shot — citations are always correct this way.
            if agent_type == "agentic":
                yield f"event: status\ndata: {json.dumps({'status': 'Thinking...'})}\n\n"

                try:
                    agent = create_supervisor_agent(
                        project_id=project_id,
                        model="gpt-4o",
                        chat_history=chat_history
                    )
                    yield f"event: status\ndata: {json.dumps({'status': 'Searching documents...'})}\n\n"

                    result = await asyncio.to_thread(agent.invoke, {
                        "messages": [{"role": "user", "content": message_content}]
                    })
                except Exception as agent_error:
                    logger.error("agent_error", chat_id=chat_id, error=str(agent_error), exc_info=True)
                    yield f"event: error\ndata: {json.dumps({'message': str(agent_error)})}\n\n"
                    return

                full_response = result["messages"][-1].content
                citations = result.get("citations", [])

                logger.info("agent_invocation_completed",
                           chat_id=chat_id,
                           response_length=len(full_response),
                           citations_count=len(citations))

                # Stream word by word to give streaming feel
                # Citations are already captured correctly from invoke()
                words = full_response.split(" ")
                for i, word in enumerate(words):
                    chunk = word if i == len(words) - 1 else word + " "
                    yield f"event: token\ndata: {json.dumps({'content': chunk})}\n\n"
                    await asyncio.sleep(0.02)  # 20ms delay per word — adjust to taste

            # ── SIMPLE agent: true token-by-token streaming ───────────────────
            else:
                agent = create_simple_rag_agent(
                    project_id=project_id,
                    model="gpt-4o",
                    chat_history=chat_history
                )

                full_response = ""
                citations = []
                guardrail_node_done = False
                in_guardrail_llm = False
                tool_called = False
                is_final_response = False

                async for event in agent.astream_events(
                    {"messages": [{"role": "user", "content": message_content}]},
                    version="v2"
                ):
                    kind = event["event"]
                    name = event.get("name", "")

                    # Track guardrail LLM start — skip its tokens
                    if kind == "on_chat_model_start" and not guardrail_node_done:
                        in_guardrail_llm = True

                    # Guardrail node completion
                    elif kind == "on_chain_end" and name == "guardrail":
                        output = event.get("data", {}).get("output", {})
                        in_guardrail_llm = False
                        if output.get("guardrail_passed") == False:
                            messages = output.get("messages", [])
                            if messages:
                                rejection_content = (
                                    messages[0].content
                                    if hasattr(messages[0], "content")
                                    else str(messages[0])
                                )
                                full_response = rejection_content
                                yield f"event: token\ndata: {json.dumps({'content': rejection_content})}\n\n"
                        else:
                            guardrail_node_done = True
                            yield f"event: status\ndata: {json.dumps({'status': 'Thinking...'})}\n\n"

                    # Tool start
                    elif kind == "on_tool_start":
                        tool_called = True
                        if name == "rag_search":
                            yield f"event: status\ndata: {json.dumps({'status': 'Searching documents...'})}\n\n"
                        else:
                            yield f"event: status\ndata: {json.dumps({'status': 'Working...'})}\n\n"

                    # rag_search tool end — capture citations from tool output directly
                    elif kind == "on_tool_end" and name == "rag_search":
                        tool_output = event.get("data", {}).get("output")
                        if tool_output is not None:
                            if hasattr(tool_output, "update") and isinstance(tool_output.update, dict):
                                new_citations = tool_output.update.get("citations", [])
                            elif isinstance(tool_output, dict):
                                new_citations = tool_output.get("citations", [])
                            else:
                                new_citations = []
                            if new_citations:
                                citations.extend(new_citations)
                                logger.info("citations_captured_from_tool",
                                           count=len(new_citations),
                                           total=len(citations))

                        is_final_response = True
                        yield f"event: status\ndata: {json.dumps({'status': 'Generating response...'})}\n\n"

                    # Other tool end
                    elif kind == "on_tool_end":
                        is_final_response = True
                        yield f"event: status\ndata: {json.dumps({'status': 'Generating response...'})}\n\n"

                    # Token streaming
                    elif kind == "on_chat_model_stream":
                        if guardrail_node_done and not in_guardrail_llm and (is_final_response or not tool_called):
                            chunk = event["data"].get("chunk")
                            if chunk:
                                content = chunk.content if hasattr(chunk, "content") else ""
                                if content:
                                    full_response += content
                                    yield f"event: token\ndata: {json.dumps({'content': content})}\n\n"

                logger.info("agent_invocation_completed",
                           chat_id=chat_id,
                           response_length=len(full_response),
                           citations_count=len(citations))

            # Step 6: Insert AI response into database
            ai_response_insert_data = {
                "content": full_response,
                "chat_id": chat_id,
                "clerk_id": clerk_id,
                "role": MessageRole.ASSISTANT.value,
                "citations": citations,
            }
            ai_response_creation_result = (
                supabase.table("messages").insert(ai_response_insert_data).execute()
            )
            if not ai_response_creation_result.data:
                logger.error("ai_response_creation_failed", chat_id=chat_id, reason="no_data_returned")
                yield f"event: error\ndata: {json.dumps({'message': 'Failed to save AI response'})}\n\n"
                return

            ai_message_data = ai_response_creation_result.data[0]
            logger.info("message_sent_successfully", chat_id=chat_id, ai_message_id=ai_message_data["id"])

            # Step 7: Done event
            yield f"event: done\ndata: {json.dumps({'userMessage': user_message_data, 'aiMessage': ai_message_data})}\n\n"

        except Exception as e:
            logger.error("stream_send_message_error", chat_id=chat_id, error=str(e), exc_info=True)
            yield f"event: error\ndata: {json.dumps({'message': str(e)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )
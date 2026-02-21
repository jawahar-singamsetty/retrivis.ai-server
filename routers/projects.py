from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from auth import get_current_user
from database import supabase

router = APIRouter(
    tags = ["projects"]
)


class ProjectCreate(BaseModel):
    name: str
    description: str = ""


@router.get("/api/projects")
def get_projects(clerk_id: str = Depends(get_current_user)):
    try:
        result = supabase.table('projects').select('*').eq('clerk_id', clerk_id).execute()
        return {
            "message": "Projects fetched successfully",
            "data": result.data
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch projects: {str(e)}")



@router.post("/api/projects")
def create_project(project: ProjectCreate, clerk_id: str = Depends(get_current_user)):
    try:
        # Insert new project into database
        project_result = supabase.table('projects').insert({
            "name": project.name,
            "description": project.description,
            "clerk_id": clerk_id
        }).execute()
        
        if not project_result.data:
            raise HTTPException(status_code=500, detail="Failed to create project")

        created_project = project_result.data[0]
        project_id = created_project['id']

        # Create default project settings
        settings_result = supabase.table('project_settings').insert({
            "project_id": project_id,
            "embedding_model": "text-embedding-3-large",
            "rag_strategy": "basic",
            "agent_type": "agentic",
            "chunks_per_search": 10,
            "final_context_size": 5,
            "similarity_threshold": 0.3,
            "number_of_queries": 5,
            "reranking_enabled": True,
            "reranking_model": "rerank-english-v3.0",
            "vector_weight": 0.7,
            "keyword_weight": 0.3,
        }).execute()

        if not settings_result.data:
            # if settings creation fails, delete the project
            supabase.table('projects').delete().eq('id', project_id).execute()
            raise HTTPException(status_code=500, detail="Failed to create project settings")
        
        return {
            "message": "Project created successfully",
            "data": created_project
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create project: {str(e)}")

    
@router.delete("/api/projects/{project_id}")
def delete_project(
    project_id: str, 
    clerk_id: str = Depends(get_current_user)
):
    try:
        # Verify project exists and belongs to the user
        project_result = supabase.table('projects').select('*').eq('id', project_id).eq('clerk_id', clerk_id).execute()
        
        if not project_result.data:
            raise HTTPException(status_code=404, detail="Project not found or you don't have permission to delete it")
        
        # Delete project and associated settings
        
        deleted_result = supabase.table('projects').delete().eq('id', project_id).eq('clerk_id', clerk_id).execute()
        
        if not deleted_result.data:
            raise HTTPException(status_code=500, detail="Failed to delete project")
        
        return {
            "message": "Project deleted successfully",
            "data": deleted_result.data[0]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete project: {str(e)}")
    
    
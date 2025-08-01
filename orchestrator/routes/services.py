"""Service registry routes."""

from typing import List
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel

from shared_libs.auth import get_current_user
from shared_libs.models import ServiceRegistration


router = APIRouter()


class ServiceRegisterRequest(BaseModel):
    name: str
    url: str
    health_endpoint: str = "/health"
    metadata: dict = {}


@router.post("/register", response_model=dict)
async def register_service(
    service_data: ServiceRegisterRequest,
    current_user: dict = Depends(get_current_user)
):
    """Register a new service."""
    from main import app
    service_registry = app.state.service_registry
    
    service = ServiceRegistration(
        name=service_data.name,
        url=service_data.url,
        health_endpoint=service_data.health_endpoint,
        metadata=service_data.metadata
    )
    
    success = await service_registry.register_service(service)
    
    if not success:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Failed to register service"
        )
    
    return {"message": f"Service {service_data.name} registered successfully"}


@router.delete("/{service_name}", response_model=dict)
async def unregister_service(
    service_name: str,
    current_user: dict = Depends(get_current_user)
):
    """Unregister a service."""
    from main import app
    service_registry = app.state.service_registry
    
    success = await service_registry.unregister_service(service_name)
    
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Service not found"
        )
    
    return {"message": f"Service {service_name} unregistered successfully"}


@router.get("/", response_model=List[ServiceRegistration])
async def list_services(current_user: dict = Depends(get_current_user)):
    """List all registered services."""
    from main import app
    service_registry = app.state.service_registry
    
    services = await service_registry.get_all_services()
    return services


@router.get("/{service_name}", response_model=ServiceRegistration)
async def get_service(
    service_name: str,
    current_user: dict = Depends(get_current_user)
):
    """Get a specific service."""
    from main import app
    service_registry = app.state.service_registry
    
    service = await service_registry.get_service(service_name)
    
    if not service:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Service not found"
        )
    
    return service


@router.post("/{service_name}/heartbeat", response_model=dict)
async def service_heartbeat(service_name: str):
    """Update service heartbeat."""
    from main import app
    service_registry = app.state.service_registry
    
    success = await service_registry.update_heartbeat(service_name)
    
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Service not found"
        )
    
    return {"message": "Heartbeat updated"}


services_router = router

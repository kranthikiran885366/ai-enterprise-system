"""API proxy routes for forwarding requests to microservices."""

import httpx
from fastapi import APIRouter, Request, HTTPException, Depends
from fastapi.responses import Response
from loguru import logger

from shared_libs.auth import get_current_user


router = APIRouter()


@router.api_route("/{service_name}/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
async def proxy_request(
    service_name: str,
    path: str,
    request: Request,
    current_user: dict = Depends(get_current_user)
):
    """Proxy requests to microservices."""
    from main import app
    service_registry = app.state.service_registry
    
    # Get service from registry
    service = await service_registry.get_service(f"{service_name}-agent")
    
    if not service:
        raise HTTPException(status_code=404, detail=f"Service {service_name} not found")
    
    # Build target URL
    target_url = f"{service.url}/api/{service_name}/{path}"
    
    # Get request body
    body = await request.body()
    
    # Forward headers (excluding host)
    headers = dict(request.headers)
    headers.pop("host", None)
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.request(
                method=request.method,
                url=target_url,
                headers=headers,
                content=body,
                timeout=30.0
            )
            
            # Forward response
            return Response(
                content=response.content,
                status_code=response.status_code,
                headers=dict(response.headers)
            )
            
    except httpx.RequestError as e:
        logger.error(f"Error proxying request to {target_url}: {e}")
        raise HTTPException(status_code=503, detail=f"Service {service_name} unavailable")


proxy_router = router

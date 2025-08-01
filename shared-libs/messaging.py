"""Message broker utilities for inter-service communication."""

import json
import os
import asyncio
from typing import Dict, Any, Callable, Optional
import pika
import pika.adapters.asyncio_connection
from loguru import logger


class MessageBroker:
    """RabbitMQ message broker for inter-service communication."""
    
    def __init__(self):
        self.connection: Optional[pika.adapters.asyncio_connection.AsyncioConnection] = None
        self.channel: Optional[pika.adapters.asyncio_connection.AsyncioChannel] = None
        self.rabbitmq_url = os.getenv("RABBITMQ_URL", "amqp://localhost:5672/")
        self.callbacks: Dict[str, Callable] = {}
    
    async def connect(self):
        """Connect to RabbitMQ."""
        try:
            parameters = pika.URLParameters(self.rabbitmq_url)
            self.connection = await pika.adapters.asyncio_connection.AsyncioConnection.create(
                parameters
            )
            self.channel = await self.connection.channel()
            logger.info("Connected to RabbitMQ")
        except Exception as e:
            logger.error(f"Failed to connect to RabbitMQ: {e}")
            raise
    
    async def disconnect(self):
        """Disconnect from RabbitMQ."""
        if self.connection and not self.connection.is_closed:
            await self.connection.close()
            logger.info("Disconnected from RabbitMQ")
    
    async def declare_queue(self, queue_name: str, durable: bool = True):
        """Declare a queue."""
        if not self.channel:
            raise RuntimeError("Not connected to RabbitMQ")
        
        await self.channel.queue_declare(queue=queue_name, durable=durable)
        logger.info(f"Queue declared: {queue_name}")
    
    async def publish_message(self, queue_name: str, message: Dict[str, Any]):
        """Publish a message to a queue."""
        if not self.channel:
            raise RuntimeError("Not connected to RabbitMQ")
        
        message_body = json.dumps(message)
        await self.channel.basic_publish(
            exchange="",
            routing_key=queue_name,
            body=message_body,
            properties=pika.BasicProperties(delivery_mode=2)  # Make message persistent
        )
        
        logger.info(f"Message published to {queue_name}: {message}")
    
    async def consume_messages(self, queue_name: str, callback: Callable):
        """Start consuming messages from a queue."""
        if not self.channel:
            raise RuntimeError("Not connected to RabbitMQ")
        
        self.callbacks[queue_name] = callback
        
        async def message_handler(channel, method, properties, body):
            try:
                message = json.loads(body.decode())
                await callback(message)
                await channel.basic_ack(method.delivery_tag)
            except Exception as e:
                logger.error(f"Error processing message: {e}")
                await channel.basic_nack(method.delivery_tag, requeue=False)
        
        await self.channel.basic_consume(queue_name, message_handler)
        logger.info(f"Started consuming messages from {queue_name}")
    
    async def send_to_service(self, service_name: str, action: str, data: Dict[str, Any]):
        """Send a message to a specific service."""
        queue_name = f"{service_name}_queue"
        message = {
            "action": action,
            "data": data,
            "timestamp": str(asyncio.get_event_loop().time())
        }
        await self.publish_message(queue_name, message)


# Global message broker instance
message_broker = MessageBroker()


async def get_message_broker() -> MessageBroker:
    """Dependency to get message broker instance."""
    return message_broker

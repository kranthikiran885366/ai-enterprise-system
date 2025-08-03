// MongoDB initialization script for AI Enterprise System

// Switch to enterprise database
use enterprise;

// Create collections with validation schemas

// Employees collection
db.createCollection("employees", {
  validator: {
    $jsonSchema: {
      bsonType: "object",
      required: ["employee_id", "first_name", "last_name", "email", "department", "position", "hire_date"],
      properties: {
        employee_id: { bsonType: "string" },
        first_name: { bsonType: "string" },
        last_name: { bsonType: "string" },
        email: { bsonType: "string" },
        department: { enum: ["engineering", "sales", "marketing", "hr", "finance", "legal", "admin", "it", "support"] },
        position: { bsonType: "string" },
        hire_date: { bsonType: "date" },
        salary: { bsonType: "number", minimum: 0 },
        status: { enum: ["active", "inactive", "terminated", "on_leave"] }
      }
    }
  }
});

// Expenses collection
db.createCollection("expenses", {
  validator: {
    $jsonSchema: {
      bsonType: "object",
      required: ["expense_id", "employee_id", "amount", "category", "description", "expense_date"],
      properties: {
        expense_id: { bsonType: "string" },
        employee_id: { bsonType: "string" },
        amount: { bsonType: "number", minimum: 0 },
        category: { enum: ["travel", "office_supplies", "marketing", "utilities", "software", "equipment", "meals", "other"] },
        description: { bsonType: "string" },
        expense_date: { bsonType: "date" },
        status: { enum: ["pending", "approved", "rejected", "paid"] }
      }
    }
  }
});

// Leads collection
db.createCollection("leads", {
  validator: {
    $jsonSchema: {
      bsonType: "object",
      required: ["lead_id", "company_name", "contact_name", "contact_email", "source"],
      properties: {
        lead_id: { bsonType: "string" },
        company_name: { bsonType: "string" },
        contact_name: { bsonType: "string" },
        contact_email: { bsonType: "string" },
        source: { enum: ["website", "referral", "cold_outreach", "social_media", "event", "advertisement", "partner"] },
        status: { enum: ["new", "contacted", "qualified", "proposal", "negotiation", "closed_won", "closed_lost"] },
        ai_score: { bsonType: "number", minimum: 0, maximum: 1 }
      }
    }
  }
});

// Support tickets collection
db.createCollection("tickets", {
  validator: {
    $jsonSchema: {
      bsonType: "object",
      required: ["ticket_id", "customer_email", "customer_name", "subject", "description", "category"],
      properties: {
        ticket_id: { bsonType: "string" },
        customer_email: { bsonType: "string" },
        customer_name: { bsonType: "string" },
        subject: { bsonType: "string" },
        description: { bsonType: "string" },
        category: { enum: ["technical", "billing", "feature_request", "bug_report", "general"] },
        priority: { enum: ["low", "medium", "high", "critical"] },
        status: { enum: ["open", "in_progress", "waiting_customer", "resolved", "closed"] }
      }
    }
  }
});

// IT Assets collection
db.createCollection("it_assets", {
  validator: {
    $jsonSchema: {
      bsonType: "object",
      required: ["asset_id", "asset_tag", "name", "asset_type"],
      properties: {
        asset_id: { bsonType: "string" },
        asset_tag: { bsonType: "string" },
        name: { bsonType: "string" },
        asset_type: { enum: ["laptop", "desktop", "server", "network_device", "mobile_device", "software_license", "printer", "monitor"] },
        status: { enum: ["active", "inactive", "maintenance", "retired", "lost", "stolen"] }
      }
    }
  }
});

// Contracts collection
db.createCollection("contracts", {
  validator: {
    $jsonSchema: {
      bsonType: "object",
      required: ["contract_id", "title", "contract_type"],
      properties: {
        contract_id: { bsonType: "string" },
        title: { bsonType: "string" },
        contract_type: { enum: ["employment", "vendor", "client", "nda", "service_agreement", "lease", "partnership"] },
        status: { enum: ["draft", "under_review", "pending_signature", "active", "expired", "terminated"] }
      }
    }
  }
});

// Marketing campaigns collection
db.createCollection("campaigns", {
  validator: {
    $jsonSchema: {
      bsonType: "object",
      required: ["campaign_id", "name", "campaign_type", "created_by"],
      properties: {
        campaign_id: { bsonType: "string" },
        name: { bsonType: "string" },
        campaign_type: { enum: ["email", "social_media", "content", "paid_ads", "webinar", "event"] },
        status: { enum: ["draft", "scheduled", "active", "paused", "completed", "cancelled"] },
        budget: { bsonType: "number", minimum: 0 }
      }
    }
  }
});

// Business rules collection
db.createCollection("rules", {
  validator: {
    $jsonSchema: {
      bsonType: "object",
      required: ["rule_id", "name", "rule_type", "conditions", "actions", "created_by"],
      properties: {
        rule_id: { bsonType: "string" },
        name: { bsonType: "string" },
        rule_type: { enum: ["threshold", "condition", "pattern", "anomaly"] },
        status: { enum: ["active", "inactive", "testing"] },
        priority: { bsonType: "int", minimum: 1, maximum: 10 }
      }
    }
  }
});

// Create indexes for better performance
print("Creating indexes...");

// Employee indexes
db.employees.createIndex({ "employee_id": 1 }, { unique: true });
db.employees.createIndex({ "email": 1 }, { unique: true });
db.employees.createIndex({ "department": 1, "status": 1 });
db.employees.createIndex({ "manager_id": 1 });
db.employees.createIndex({ "hire_date": 1 });

// Expense indexes
db.expenses.createIndex({ "expense_id": 1 }, { unique: true });
db.expenses.createIndex({ "employee_id": 1, "expense_date": -1 });
db.expenses.createIndex({ "status": 1 });
db.expenses.createIndex({ "category": 1 });
db.expenses.createIndex({ "amount": 1 });

// Lead indexes
db.leads.createIndex({ "lead_id": 1 }, { unique: true });
db.leads.createIndex({ "contact_email": 1 }, { unique: true });
db.leads.createIndex({ "status": 1 });
db.leads.createIndex({ "assigned_to": 1 });
db.leads.createIndex({ "ai_score": -1 });
db.leads.createIndex({ "source": 1 });

// Ticket indexes
db.tickets.createIndex({ "ticket_id": 1 }, { unique: true });
db.tickets.createIndex({ "customer_email": 1 });
db.tickets.createIndex({ "status": 1 });
db.tickets.createIndex({ "priority": 1 });
db.tickets.createIndex({ "category": 1 });
db.tickets.createIndex({ "assigned_to": 1 });
db.tickets.createIndex({ "created_at": -1 });

// Asset indexes
db.it_assets.createIndex({ "asset_id": 1 }, { unique: true });
db.it_assets.createIndex({ "asset_tag": 1 }, { unique: true });
db.it_assets.createIndex({ "assigned_to": 1 });
db.it_assets.createIndex({ "status": 1 });
db.it_assets.createIndex({ "asset_type": 1 });

// Contract indexes
db.contracts.createIndex({ "contract_id": 1 }, { unique: true });
db.contracts.createIndex({ "contract_type": 1 });
db.contracts.createIndex({ "status": 1 });
db.contracts.createIndex({ "end_date": 1 });
db.contracts.createIndex({ "assigned_lawyer": 1 });

// Campaign indexes
db.campaigns.createIndex({ "campaign_id": 1 }, { unique: true });
db.campaigns.createIndex({ "campaign_type": 1 });
db.campaigns.createIndex({ "status": 1 });
db.campaigns.createIndex({ "created_by": 1 });
db.campaigns.createIndex({ "start_date": 1 });

// Rule indexes
db.rules.createIndex({ "rule_id": 1 }, { unique: true });
db.rules.createIndex({ "status": 1 });
db.rules.createIndex({ "department": 1 });
db.rules.createIndex({ "priority": -1 });

// Data lake indexes
db.data_lake_events.createIndex({ "timestamp": -1 });
db.data_lake_events.createIndex({ "agent": 1, "event_type": 1 });
db.data_lake_events.createIndex({ "entity_type": 1, "entity_id": 1 });

db.data_lake_metrics.createIndex({ "date": -1 });
db.data_lake_metrics.createIndex({ "agent": 1, "metric_name": 1 });

// Service registry indexes
db.service_registry.createIndex({ "name": 1 }, { unique: true });
db.service_registry.createIndex({ "status": 1 });
db.service_registry.createIndex({ "last_heartbeat": -1 });

// User and auth indexes
db.users.createIndex({ "username": 1 }, { unique: true });
db.users.createIndex({ "email": 1 }, { unique: true });

print("MongoDB initialization completed successfully!");
print("Collections created with validation schemas and indexes.");
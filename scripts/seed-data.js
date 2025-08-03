// Seed data script for AI Enterprise System

use enterprise;

print("Seeding initial data...");

// Seed employees
db.employees.insertMany([
  {
    employee_id: "EMP001",
    first_name: "John",
    last_name: "Doe",
    email: "john.doe@company.com",
    phone: "+1-555-0101",
    department: "engineering",
    position: "Senior Software Engineer",
    manager_id: "EMP005",
    hire_date: new Date("2023-01-15"),
    salary: 95000,
    status: "active",
    skills: ["Python", "JavaScript", "React", "Node.js"],
    created_at: new Date(),
    updated_at: new Date()
  },
  {
    employee_id: "EMP002",
    first_name: "Jane",
    last_name: "Smith",
    email: "jane.smith@company.com",
    phone: "+1-555-0102",
    department: "sales",
    position: "Sales Manager",
    hire_date: new Date("2022-08-20"),
    salary: 85000,
    status: "active",
    skills: ["Sales", "CRM", "Negotiation"],
    created_at: new Date(),
    updated_at: new Date()
  },
  {
    employee_id: "EMP003",
    first_name: "Mike",
    last_name: "Johnson",
    email: "mike.johnson@company.com",
    phone: "+1-555-0103",
    department: "marketing",
    position: "Marketing Specialist",
    hire_date: new Date("2023-03-10"),
    salary: 65000,
    status: "active",
    skills: ["Digital Marketing", "SEO", "Content Creation"],
    created_at: new Date(),
    updated_at: new Date()
  },
  {
    employee_id: "EMP004",
    first_name: "Sarah",
    last_name: "Wilson",
    email: "sarah.wilson@company.com",
    phone: "+1-555-0104",
    department: "hr",
    position: "HR Manager",
    hire_date: new Date("2022-05-01"),
    salary: 75000,
    status: "active",
    skills: ["Recruitment", "Employee Relations", "Policy Development"],
    created_at: new Date(),
    updated_at: new Date()
  },
  {
    employee_id: "EMP005",
    first_name: "David",
    last_name: "Brown",
    email: "david.brown@company.com",
    phone: "+1-555-0105",
    department: "engineering",
    position: "Engineering Director",
    hire_date: new Date("2021-01-15"),
    salary: 140000,
    status: "active",
    skills: ["Leadership", "Architecture", "Strategy"],
    created_at: new Date(),
    updated_at: new Date()
  }
]);

// Seed sample expenses
db.expenses.insertMany([
  {
    expense_id: "EXP001",
    employee_id: "EMP001",
    amount: 125.50,
    currency: "USD",
    category: "meals",
    description: "Client lunch meeting",
    expense_date: new Date("2024-01-10"),
    status: "approved",
    receipt_url: "https://example.com/receipt1.pdf",
    created_at: new Date(),
    updated_at: new Date()
  },
  {
    expense_id: "EXP002",
    employee_id: "EMP002",
    amount: 450.00,
    currency: "USD",
    category: "travel",
    description: "Flight to customer site",
    expense_date: new Date("2024-01-12"),
    status: "pending",
    receipt_url: "https://example.com/receipt2.pdf",
    created_at: new Date(),
    updated_at: new Date()
  },
  {
    expense_id: "EXP003",
    employee_id: "EMP003",
    amount: 89.99,
    currency: "USD",
    category: "software",
    description: "Design software subscription",
    expense_date: new Date("2024-01-08"),
    status: "approved",
    created_at: new Date(),
    updated_at: new Date()
  }
]);

// Seed sample leads
db.leads.insertMany([
  {
    lead_id: "LEAD001",
    company_name: "TechCorp Inc",
    contact_name: "Alice Johnson",
    contact_email: "alice@techcorp.com",
    contact_phone: "+1-555-0201",
    job_title: "CTO",
    company_size: 250,
    industry: "technology",
    budget: 50000,
    source: "website",
    status: "qualified",
    ai_score: 0.85,
    assigned_to: "EMP002",
    tags: ["enterprise", "high-value"],
    created_at: new Date(),
    updated_at: new Date()
  },
  {
    lead_id: "LEAD002",
    company_name: "StartupXYZ",
    contact_name: "Bob Chen",
    contact_email: "bob@startupxyz.com",
    contact_phone: "+1-555-0202",
    job_title: "Founder",
    company_size: 15,
    industry: "fintech",
    budget: 15000,
    source: "referral",
    status: "new",
    ai_score: 0.65,
    tags: ["startup", "fintech"],
    created_at: new Date(),
    updated_at: new Date()
  }
]);

// Seed sample support tickets
db.tickets.insertMany([
  {
    ticket_id: "TICK001",
    customer_id: "CUST001",
    customer_email: "customer1@example.com",
    customer_name: "Robert Davis",
    subject: "Login issues with mobile app",
    description: "Unable to login to mobile app after recent update. Getting error message 'Invalid credentials' even with correct password.",
    category: "technical",
    priority: "medium",
    status: "open",
    customer_tier: "premium",
    tags: ["mobile", "login", "authentication"],
    created_at: new Date(),
    updated_at: new Date()
  },
  {
    ticket_id: "TICK002",
    customer_id: "CUST002",
    customer_email: "customer2@example.com",
    customer_name: "Lisa Garcia",
    subject: "Billing discrepancy",
    description: "My invoice shows charges that don't match my subscription plan. Need clarification on additional fees.",
    category: "billing",
    priority: "high",
    status: "in_progress",
    customer_tier: "enterprise",
    assigned_to: "support_agent_1",
    tags: ["billing", "invoice", "subscription"],
    created_at: new Date(),
    updated_at: new Date()
  }
]);

// Seed sample IT assets
db.it_assets.insertMany([
  {
    asset_id: "AST001",
    asset_tag: "TAG001",
    name: "MacBook Pro 16-inch",
    asset_type: "laptop",
    brand: "Apple",
    model: "MacBook Pro",
    serial_number: "C02XD0AAHV29",
    purchase_date: new Date("2023-06-15"),
    purchase_cost: 2499.00,
    warranty_expiry: new Date("2026-06-15"),
    assigned_to: "EMP001",
    location: "Office Floor 2",
    status: "active",
    specifications: {
      processor: "M2 Pro",
      memory: "16GB",
      storage: "512GB SSD"
    },
    created_at: new Date(),
    updated_at: new Date()
  },
  {
    asset_id: "AST002",
    asset_tag: "TAG002",
    name: "Dell OptiPlex 7090",
    asset_type: "desktop",
    brand: "Dell",
    model: "OptiPlex 7090",
    serial_number: "DELL123456",
    purchase_date: new Date("2023-04-20"),
    purchase_cost: 1299.00,
    warranty_expiry: new Date("2026-04-20"),
    assigned_to: "EMP003",
    location: "Office Floor 1",
    status: "active",
    specifications: {
      processor: "Intel i7",
      memory: "16GB",
      storage: "256GB SSD"
    },
    created_at: new Date(),
    updated_at: new Date()
  }
]);

// Seed sample contracts
db.contracts.insertMany([
  {
    contract_id: "CONT001",
    title: "Software Development Services Agreement",
    contract_type: "client",
    status: "active",
    parties: [
      { name: "Our Company", role: "service_provider" },
      { name: "Client Corp", role: "client" }
    ],
    start_date: new Date("2024-01-01"),
    end_date: new Date("2024-12-31"),
    value: 120000,
    currency: "USD",
    assigned_lawyer: "legal_team@company.com",
    risk_score: 0.3,
    created_at: new Date(),
    updated_at: new Date()
  },
  {
    contract_id: "CONT002",
    title: "Office Lease Agreement",
    contract_type: "lease",
    status: "active",
    parties: [
      { name: "Our Company", role: "tenant" },
      { name: "Property Management LLC", role: "landlord" }
    ],
    start_date: new Date("2023-01-01"),
    end_date: new Date("2025-12-31"),
    renewal_date: new Date("2025-10-01"),
    auto_renewal: true,
    value: 240000,
    currency: "USD",
    created_at: new Date(),
    updated_at: new Date()
  }
]);

// Seed sample campaigns
db.campaigns.insertMany([
  {
    campaign_id: "CAMP001",
    name: "Q1 Product Launch Campaign",
    description: "Email campaign for new product launch",
    campaign_type: "email",
    status: "active",
    start_date: new Date("2024-01-01"),
    end_date: new Date("2024-03-31"),
    budget: 15000,
    target_audience: {
      size: 5000,
      segments: ["existing_customers", "qualified_leads"]
    },
    goals: ["increase_awareness", "generate_leads", "drive_sales"],
    created_by: "EMP003",
    created_at: new Date(),
    updated_at: new Date()
  },
  {
    campaign_id: "CAMP002",
    name: "Social Media Brand Awareness",
    description: "Multi-platform social media campaign",
    campaign_type: "social_media",
    status: "scheduled",
    start_date: new Date("2024-02-01"),
    end_date: new Date("2024-04-30"),
    budget: 8000,
    target_audience: {
      size: 10000,
      segments: ["tech_professionals", "decision_makers"]
    },
    goals: ["brand_awareness", "engagement"],
    created_by: "EMP003",
    created_at: new Date(),
    updated_at: new Date()
  }
]);

// Seed sample business rules
db.rules.insertMany([
  {
    rule_id: "RULE001",
    name: "High Expense Alert",
    description: "Alert when expense exceeds $1000",
    rule_type: "threshold",
    conditions: {
      field: "expense.amount",
      operator: "greater_than",
      value: 1000
    },
    actions: [
      {
        type: "alert",
        severity: "high",
        message: "High expense detected: ${amount}"
      },
      {
        type: "notification",
        recipient: "finance_manager",
        message: "Expense requires immediate review"
      }
    ],
    priority: 8,
    status: "active",
    department: "finance",
    created_by: "system",
    trigger_count: 0,
    created_at: new Date(),
    updated_at: new Date()
  },
  {
    rule_id: "RULE002",
    name: "New Employee Onboarding",
    description: "Trigger onboarding workflow for new employees",
    rule_type: "condition",
    conditions: {
      field: "employee.status",
      operator: "equals",
      value: "new"
    },
    actions: [
      {
        type: "automation",
        action: "start_onboarding_workflow",
        parameters: { employee_id: "{employee.id}" }
      },
      {
        type: "notification",
        recipient: "hr_team",
        message: "New employee onboarding required"
      }
    ],
    priority: 9,
    status: "active",
    department: "hr",
    created_by: "system",
    trigger_count: 0,
    created_at: new Date(),
    updated_at: new Date()
  },
  {
    rule_id: "RULE003",
    name: "Critical Support Ticket Escalation",
    description: "Auto-escalate critical priority tickets",
    rule_type: "condition",
    conditions: {
      field: "ticket.priority",
      operator: "equals",
      value: "critical"
    },
    actions: [
      {
        type: "escalation",
        level: "senior_support",
        department: "support"
      },
      {
        type: "notification",
        recipient: "support_manager",
        message: "Critical ticket requires immediate attention"
      }
    ],
    priority: 10,
    status: "active",
    department: "support",
    created_by: "system",
    trigger_count: 0,
    created_at: new Date(),
    updated_at: new Date()
  }
]);

// Seed sample announcements
db.announcements.insertMany([
  {
    announcement_id: "ANN001",
    title: "New Company Policy: Remote Work Guidelines",
    content: "We are pleased to announce our new remote work policy that provides greater flexibility for all employees. The policy includes guidelines for home office setup, communication expectations, and performance metrics. Please review the full policy document in the employee portal.",
    announcement_type: "policy_update",
    priority: "high",
    target_audience: ["all"],
    author: "EMP004",
    published: true,
    publish_date: new Date(),
    acknowledgment_required: true,
    read_by: ["EMP001", "EMP002"],
    acknowledged_by: ["EMP001"],
    created_at: new Date(),
    updated_at: new Date()
  },
  {
    announcement_id: "ANN002",
    title: "System Maintenance Scheduled",
    content: "Scheduled system maintenance will occur this Saturday from 2:00 AM to 6:00 AM EST. All systems will be temporarily unavailable during this time. Please plan accordingly and save your work before the maintenance window.",
    announcement_type: "system_maintenance",
    priority: "medium",
    target_audience: ["all"],
    author: "it_admin",
    published: true,
    publish_date: new Date(),
    expiry_date: new Date(Date.now() + 7 * 24 * 60 * 60 * 1000), // 7 days from now
    created_at: new Date(),
    updated_at: new Date()
  }
]);

// Seed sample policies
db.policies.insertMany([
  {
    policy_id: "POL001",
    title: "Remote Work Policy",
    description: "Guidelines and requirements for remote work arrangements",
    policy_type: "hr_policy",
    content: "This policy outlines the company's approach to remote work, including eligibility criteria, equipment provision, communication expectations, and performance management for remote employees.",
    version: "1.0",
    status: "active",
    effective_date: new Date("2024-01-01"),
    review_date: new Date("2024-12-31"),
    author: "EMP004",
    approver: "hr_director",
    approval_date: new Date("2023-12-15"),
    applicable_departments: ["all"],
    compliance_requirements: ["employment_law", "data_security"],
    created_at: new Date(),
    updated_at: new Date()
  },
  {
    policy_id: "POL002",
    title: "Information Security Policy",
    description: "Security requirements and best practices for all employees",
    policy_type: "security_policy",
    content: "This policy establishes security requirements for accessing company systems, handling confidential information, and reporting security incidents.",
    version: "2.1",
    status: "active",
    effective_date: new Date("2023-06-01"),
    review_date: new Date("2024-06-01"),
    author: "it_security",
    approver: "ciso",
    approval_date: new Date("2023-05-15"),
    applicable_departments: ["all"],
    compliance_requirements: ["iso27001", "gdpr"],
    created_at: new Date(),
    updated_at: new Date()
  }
]);

// Seed sample deals
db.deals.insertMany([
  {
    deal_id: "DEAL001",
    lead_id: "LEAD001",
    company_name: "TechCorp Inc",
    contact_name: "Alice Johnson",
    contact_email: "alice@techcorp.com",
    deal_value: 75000,
    currency: "USD",
    stage: "proposal",
    probability: 70,
    expected_close_date: new Date("2024-03-15"),
    assigned_to: "EMP002",
    products: [
      { name: "Enterprise Software License", quantity: 1, price: 50000 },
      { name: "Professional Services", quantity: 100, price: 250 }
    ],
    notes: [
      {
        note: "Initial proposal sent, waiting for feedback",
        created_at: new Date(),
        created_by: "EMP002"
      }
    ],
    activities: [],
    created_at: new Date(),
    updated_at: new Date()
  }
]);

// Seed sample invoices
db.invoices.insertMany([
  {
    invoice_id: "INV001",
    client_name: "TechCorp Inc",
    client_email: "billing@techcorp.com",
    amount: 25000,
    currency: "USD",
    description: "Software development services - January 2024",
    line_items: [
      { description: "Development hours", quantity: 100, rate: 150, amount: 15000 },
      { description: "Project management", quantity: 40, rate: 125, amount: 5000 },
      { description: "Testing and QA", quantity: 50, rate: 100, amount: 5000 }
    ],
    issue_date: new Date("2024-01-31"),
    due_date: new Date("2024-02-29"),
    status: "sent",
    created_at: new Date(),
    updated_at: new Date()
  }
]);

// Seed sample knowledge articles
db.knowledge_articles.insertMany([
  {
    article_id: "KB001",
    title: "How to Reset Your Password",
    content: "To reset your password: 1. Go to the login page 2. Click 'Forgot Password' 3. Enter your email address 4. Check your email for reset instructions 5. Follow the link and create a new password",
    category: "account_help",
    tags: ["password", "login", "account"],
    author: "support_team",
    status: "published",
    views: 245,
    helpful_votes: 23,
    unhelpful_votes: 2,
    created_at: new Date(),
    updated_at: new Date(),
    last_updated: new Date()
  },
  {
    article_id: "KB002",
    title: "Troubleshooting Mobile App Issues",
    content: "Common mobile app issues and solutions: 1. App crashes - try restarting the app 2. Login problems - clear app cache 3. Sync issues - check internet connection 4. Performance issues - update to latest version",
    category: "technical",
    tags: ["mobile", "troubleshooting", "app"],
    author: "tech_support",
    status: "published",
    views: 189,
    helpful_votes: 18,
    unhelpful_votes: 1,
    created_at: new Date(),
    updated_at: new Date(),
    last_updated: new Date()
  }
]);

print("Sample data seeded successfully!");
print("Created:");
print("- 5 employees");
print("- 3 expenses");
print("- 2 leads");
print("- 2 support tickets");
print("- 2 IT assets");
print("- 2 contracts");
print("- 2 campaigns");
print("- 3 business rules");
print("- 2 announcements");
print("- 2 policies");
print("- 1 deal");
print("- 1 invoice");
print("- 2 knowledge articles");
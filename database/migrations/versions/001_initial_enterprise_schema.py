"""Initial Enterprise schema for all agents.

Revision ID: 001
Revises: 
Create Date: 2024-01-01 00:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers
revision = '001'
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Create initial enterprise schema."""
    
    # 1. Users & Authentication table
    op.create_table(
        'users',
        sa.Column('id', sa.String(36), primary_key=True),
        sa.Column('email', sa.String(255), unique=True, nullable=False),
        sa.Column('username', sa.String(255), unique=True, nullable=False),
        sa.Column('password_hash', sa.String(512), nullable=False),
        sa.Column('full_name', sa.String(255), nullable=True),
        sa.Column('department', sa.String(100), nullable=True),
        sa.Column('role', sa.String(50), nullable=False),
        sa.Column('is_active', sa.Boolean(), default=True),
        sa.Column('is_verified', sa.Boolean(), default=False),
        sa.Column('created_at', sa.DateTime(), server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(), server_default=sa.func.now(), onupdate=sa.func.now()),
    )
    op.create_index('ix_users_email', 'users', ['email'])
    op.create_index('ix_users_username', 'users', ['username'])
    
    # 2. HR Agent Tables
    # Employees
    op.create_table(
        'hr_employees',
        sa.Column('id', sa.String(36), primary_key=True),
        sa.Column('user_id', sa.String(36), sa.ForeignKey('users.id'), nullable=False),
        sa.Column('employee_id', sa.String(50), unique=True, nullable=False),
        sa.Column('first_name', sa.String(100), nullable=False),
        sa.Column('last_name', sa.String(100), nullable=False),
        sa.Column('email', sa.String(255), unique=True, nullable=False),
        sa.Column('phone', sa.String(20), nullable=True),
        sa.Column('department', sa.String(100), nullable=False),
        sa.Column('position', sa.String(100), nullable=False),
        sa.Column('salary', sa.Float(), nullable=True),
        sa.Column('employment_type', sa.String(50), nullable=False),
        sa.Column('hire_date', sa.Date(), nullable=False),
        sa.Column('status', sa.String(50), default='active'),
        sa.Column('manager_id', sa.String(36), nullable=True),
        sa.Column('created_at', sa.DateTime(), server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(), onupdate=sa.func.now()),
    )
    op.create_index('ix_hr_employees_employee_id', 'hr_employees', ['employee_id'])
    op.create_index('ix_hr_employees_email', 'hr_employees', ['email'])
    
    # Recruitment
    op.create_table(
        'hr_candidates',
        sa.Column('id', sa.String(36), primary_key=True),
        sa.Column('first_name', sa.String(100), nullable=False),
        sa.Column('last_name', sa.String(100), nullable=False),
        sa.Column('email', sa.String(255), nullable=False),
        sa.Column('phone', sa.String(20), nullable=True),
        sa.Column('position_applied', sa.String(100), nullable=False),
        sa.Column('status', sa.String(50), default='applied'),
        sa.Column('resume_url', sa.String(512), nullable=True),
        sa.Column('skills', postgresql.ARRAY(sa.String()), nullable=True),
        sa.Column('experience_years', sa.Integer(), nullable=True),
        sa.Column('ai_fit_score', sa.Float(), nullable=True),
        sa.Column('source', sa.String(100), nullable=True),
        sa.Column('created_at', sa.DateTime(), server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(), onupdate=sa.func.now()),
    )
    op.create_index('ix_hr_candidates_email', 'hr_candidates', ['email'])
    
    # Attendance
    op.create_table(
        'hr_attendance',
        sa.Column('id', sa.String(36), primary_key=True),
        sa.Column('employee_id', sa.String(36), sa.ForeignKey('hr_employees.id'), nullable=False),
        sa.Column('date', sa.Date(), nullable=False),
        sa.Column('check_in', sa.DateTime(), nullable=True),
        sa.Column('check_out', sa.DateTime(), nullable=True),
        sa.Column('status', sa.String(50), nullable=False),
        sa.Column('hours_worked', sa.Float(), nullable=True),
        sa.Column('notes', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(), server_default=sa.func.now()),
    )
    op.create_index('ix_hr_attendance_employee_date', 'hr_attendance', ['employee_id', 'date'], unique=True)
    
    # 3. Finance Agent Tables
    # Expenses
    op.create_table(
        'finance_expenses',
        sa.Column('id', sa.String(36), primary_key=True),
        sa.Column('employee_id', sa.String(36), sa.ForeignKey('hr_employees.id'), nullable=False),
        sa.Column('amount', sa.Float(), nullable=False),
        sa.Column('currency', sa.String(3), default='USD'),
        sa.Column('category', sa.String(100), nullable=False),
        sa.Column('description', sa.Text(), nullable=False),
        sa.Column('receipt_url', sa.String(512), nullable=True),
        sa.Column('status', sa.String(50), default='pending'),
        sa.Column('submitted_date', sa.DateTime(), nullable=False),
        sa.Column('approved_date', sa.DateTime(), nullable=True),
        sa.Column('approved_by', sa.String(36), nullable=True),
        sa.Column('created_at', sa.DateTime(), server_default=sa.func.now()),
    )
    op.create_index('ix_finance_expenses_employee', 'finance_expenses', ['employee_id'])
    op.create_index('ix_finance_expenses_status', 'finance_expenses', ['status'])
    
    # Invoices
    op.create_table(
        'finance_invoices',
        sa.Column('id', sa.String(36), primary_key=True),
        sa.Column('invoice_number', sa.String(50), unique=True, nullable=False),
        sa.Column('vendor_name', sa.String(255), nullable=False),
        sa.Column('vendor_id', sa.String(255), nullable=True),
        sa.Column('amount', sa.Float(), nullable=False),
        sa.Column('currency', sa.String(3), default='USD'),
        sa.Column('issue_date', sa.Date(), nullable=False),
        sa.Column('due_date', sa.Date(), nullable=False),
        sa.Column('status', sa.String(50), default='pending'),
        sa.Column('paid_date', sa.Date(), nullable=True),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(), server_default=sa.func.now()),
    )
    op.create_index('ix_finance_invoices_number', 'finance_invoices', ['invoice_number'])
    op.create_index('ix_finance_invoices_status', 'finance_invoices', ['status'])
    
    # Budget
    op.create_table(
        'finance_budgets',
        sa.Column('id', sa.String(36), primary_key=True),
        sa.Column('department', sa.String(100), nullable=False),
        sa.Column('year', sa.Integer(), nullable=False),
        sa.Column('quarter', sa.Integer(), nullable=False),
        sa.Column('allocated_amount', sa.Float(), nullable=False),
        sa.Column('spent_amount', sa.Float(), default=0),
        sa.Column('currency', sa.String(3), default='USD'),
        sa.Column('created_at', sa.DateTime(), server_default=sa.func.now()),
    )
    op.create_index('ix_finance_budgets_dept_year', 'finance_budgets', ['department', 'year'], unique=True)
    
    # Payroll
    op.create_table(
        'finance_payroll',
        sa.Column('id', sa.String(36), primary_key=True),
        sa.Column('employee_id', sa.String(36), sa.ForeignKey('hr_employees.id'), nullable=False),
        sa.Column('pay_period_start', sa.Date(), nullable=False),
        sa.Column('pay_period_end', sa.Date(), nullable=False),
        sa.Column('gross_salary', sa.Float(), nullable=False),
        sa.Column('deductions', sa.Float(), nullable=False),
        sa.Column('net_salary', sa.Float(), nullable=False),
        sa.Column('status', sa.String(50), default='pending'),
        sa.Column('paid_date', sa.Date(), nullable=True),
        sa.Column('created_at', sa.DateTime(), server_default=sa.func.now()),
    )
    op.create_index('ix_finance_payroll_employee', 'finance_payroll', ['employee_id'])
    
    # 4. Sales Agent Tables
    # Leads
    op.create_table(
        'sales_leads',
        sa.Column('id', sa.String(36), primary_key=True),
        sa.Column('company_name', sa.String(255), nullable=False),
        sa.Column('contact_name', sa.String(255), nullable=False),
        sa.Column('email', sa.String(255), nullable=False),
        sa.Column('phone', sa.String(20), nullable=True),
        sa.Column('industry', sa.String(100), nullable=True),
        sa.Column('company_size', sa.String(50), nullable=True),
        sa.Column('status', sa.String(50), default='new'),
        sa.Column('source', sa.String(100), nullable=True),
        sa.Column('assigned_to', sa.String(36), sa.ForeignKey('users.id'), nullable=True),
        sa.Column('ai_score', sa.Float(), nullable=True),
        sa.Column('budget_estimate', sa.Float(), nullable=True),
        sa.Column('timeline', sa.String(100), nullable=True),
        sa.Column('created_at', sa.DateTime(), server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(), onupdate=sa.func.now()),
    )
    op.create_index('ix_sales_leads_status', 'sales_leads', ['status'])
    op.create_index('ix_sales_leads_assigned', 'sales_leads', ['assigned_to'])
    
    # Deals
    op.create_table(
        'sales_deals',
        sa.Column('id', sa.String(36), primary_key=True),
        sa.Column('lead_id', sa.String(36), sa.ForeignKey('sales_leads.id'), nullable=False),
        sa.Column('deal_name', sa.String(255), nullable=False),
        sa.Column('amount', sa.Float(), nullable=False),
        sa.Column('currency', sa.String(3), default='USD'),
        sa.Column('stage', sa.String(50), default='prospecting'),
        sa.Column('probability', sa.Float(), nullable=True),
        sa.Column('close_date', sa.Date(), nullable=True),
        sa.Column('assigned_to', sa.String(36), sa.ForeignKey('users.id'), nullable=True),
        sa.Column('created_at', sa.DateTime(), server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(), onupdate=sa.func.now()),
    )
    op.create_index('ix_sales_deals_stage', 'sales_deals', ['stage'])
    
    # 5. Marketing Agent Tables
    op.create_table(
        'marketing_campaigns',
        sa.Column('id', sa.String(36), primary_key=True),
        sa.Column('name', sa.String(255), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('channel', sa.String(100), nullable=False),
        sa.Column('status', sa.String(50), default='draft'),
        sa.Column('budget', sa.Float(), nullable=False),
        sa.Column('spent', sa.Float(), default=0),
        sa.Column('start_date', sa.DateTime(), nullable=False),
        sa.Column('end_date', sa.DateTime(), nullable=True),
        sa.Column('impressions', sa.Integer(), default=0),
        sa.Column('clicks', sa.Integer(), default=0),
        sa.Column('conversions', sa.Integer(), default=0),
        sa.Column('created_at', sa.DateTime(), server_default=sa.func.now()),
    )
    
    # 6. Support Agent Tables
    op.create_table(
        'support_tickets',
        sa.Column('id', sa.String(36), primary_key=True),
        sa.Column('customer_name', sa.String(255), nullable=False),
        sa.Column('customer_email', sa.String(255), nullable=False),
        sa.Column('subject', sa.String(255), nullable=False),
        sa.Column('description', sa.Text(), nullable=False),
        sa.Column('priority', sa.String(50), default='medium'),
        sa.Column('status', sa.String(50), default='open'),
        sa.Column('category', sa.String(100), nullable=True),
        sa.Column('assigned_to', sa.String(36), sa.ForeignKey('users.id'), nullable=True),
        sa.Column('sentiment_score', sa.Float(), nullable=True),
        sa.Column('created_at', sa.DateTime(), server_default=sa.func.now()),
        sa.Column('resolved_at', sa.DateTime(), nullable=True),
    )
    op.create_index('ix_support_tickets_status', 'support_tickets', ['status'])
    op.create_index('ix_support_tickets_priority', 'support_tickets', ['priority'])
    
    # 7. Legal Agent Tables
    op.create_table(
        'legal_documents',
        sa.Column('id', sa.String(36), primary_key=True),
        sa.Column('document_type', sa.String(100), nullable=False),
        sa.Column('document_name', sa.String(255), nullable=False),
        sa.Column('document_url', sa.String(512), nullable=False),
        sa.Column('status', sa.String(50), default='draft'),
        sa.Column('created_date', sa.Date(), nullable=False),
        sa.Column('review_status', sa.String(50), nullable=True),
        sa.Column('risks', postgresql.ARRAY(sa.String()), nullable=True),
        sa.Column('compliance_level', sa.String(50), nullable=True),
        sa.Column('created_at', sa.DateTime(), server_default=sa.func.now()),
    )
    
    # 8. IT Agent Tables
    op.create_table(
        'it_incidents',
        sa.Column('id', sa.String(36), primary_key=True),
        sa.Column('title', sa.String(255), nullable=False),
        sa.Column('description', sa.Text(), nullable=False),
        sa.Column('service_affected', sa.String(100), nullable=False),
        sa.Column('severity', sa.String(50), default='medium'),
        sa.Column('status', sa.String(50), default='open'),
        sa.Column('assigned_to', sa.String(36), sa.ForeignKey('users.id'), nullable=True),
        sa.Column('root_cause', sa.Text(), nullable=True),
        sa.Column('resolution', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(), server_default=sa.func.now()),
        sa.Column('resolved_at', sa.DateTime(), nullable=True),
    )
    op.create_index('ix_it_incidents_status', 'it_incidents', ['status'])
    op.create_index('ix_it_incidents_severity', 'it_incidents', ['severity'])
    
    # 9. Admin Agent Tables
    op.create_table(
        'admin_audit_logs',
        sa.Column('id', sa.String(36), primary_key=True),
        sa.Column('user_id', sa.String(36), sa.ForeignKey('users.id'), nullable=False),
        sa.Column('action', sa.String(255), nullable=False),
        sa.Column('resource_type', sa.String(100), nullable=False),
        sa.Column('resource_id', sa.String(255), nullable=True),
        sa.Column('changes', postgresql.JSON(), nullable=True),
        sa.Column('ip_address', sa.String(50), nullable=True),
        sa.Column('created_at', sa.DateTime(), server_default=sa.func.now()),
    )
    op.create_index('ix_admin_audit_logs_user', 'admin_audit_logs', ['user_id'])
    op.create_index('ix_admin_audit_logs_created', 'admin_audit_logs', ['created_at'])
    
    # 10. QA Agent Tables
    op.create_table(
        'qa_test_cases',
        sa.Column('id', sa.String(36), primary_key=True),
        sa.Column('title', sa.String(255), nullable=False),
        sa.Column('description', sa.Text(), nullable=False),
        sa.Column('component', sa.String(100), nullable=False),
        sa.Column('steps', postgresql.ARRAY(sa.String()), nullable=True),
        sa.Column('expected_result', sa.Text(), nullable=True),
        sa.Column('status', sa.String(50), default='draft'),
        sa.Column('priority', sa.String(50), default='medium'),
        sa.Column('created_at', sa.DateTime(), server_default=sa.func.now()),
    )


def downgrade() -> None:
    """Drop all enterprise tables."""
    tables = [
        'qa_test_cases',
        'admin_audit_logs',
        'it_incidents',
        'legal_documents',
        'support_tickets',
        'marketing_campaigns',
        'sales_deals',
        'sales_leads',
        'finance_payroll',
        'finance_budgets',
        'finance_invoices',
        'finance_expenses',
        'hr_attendance',
        'hr_candidates',
        'hr_employees',
        'users',
    ]
    
    for table in tables:
        op.drop_table(table)

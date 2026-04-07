import axios, { AxiosInstance } from 'axios';

const apiClient: AxiosInstance = axios.create({
  baseURL: '',
  headers: { 'Content-Type': 'application/json' },
  timeout: 15000,
});

apiClient.interceptors.request.use((config) => {
  if (typeof window !== 'undefined') {
    const token = localStorage.getItem('access_token');
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
  }
  return config;
});

apiClient.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401 && typeof window !== 'undefined') {
      localStorage.removeItem('access_token');
      window.location.href = '/login';
    }
    return Promise.reject(error);
  }
);

export const authApi = {
  login: (username: string, password: string) =>
    apiClient.post('/auth/login', { username, password }),
  register: (data: { username: string; email: string; password: string }) =>
    apiClient.post('/auth/register', data),
  me: () => apiClient.get('/auth/me'),
};

export const dashboardApi = {
  getSystemStatus: () => apiClient.get('/system/status'),
  getMetrics: () => apiClient.get('/api/v1/dashboard/metrics'),
};

export const salesApi = {
  getLeads: (params?: Record<string, unknown>) => apiClient.get('/api/v1/sales/leads', { params }),
  getLead: (id: string) => apiClient.get(`/api/v1/sales/leads/${id}`),
  createLead: (data: Record<string, unknown>) => apiClient.post('/api/v1/sales/leads', data),
  updateLead: (id: string, data: Record<string, unknown>) => apiClient.put(`/api/v1/sales/leads/${id}`, data),
  getAnalytics: () => apiClient.get('/api/v1/sales/analytics'),
  getPipeline: () => apiClient.get('/api/v1/sales/pipeline'),
};

export const financeApi = {
  getExpenses: (params?: Record<string, unknown>) => apiClient.get('/api/v1/finance/expenses', { params }),
  createExpense: (data: Record<string, unknown>) => apiClient.post('/api/v1/finance/expenses', data),
  getInvoices: (params?: Record<string, unknown>) => apiClient.get('/api/v1/finance/invoices', { params }),
  getBudget: () => apiClient.get('/api/v1/finance/budget'),
  getAnalytics: () => apiClient.get('/api/v1/finance/analytics'),
};

export const hrApi = {
  getEmployees: (params?: Record<string, unknown>) => apiClient.get('/api/v1/hr/employees', { params }),
  getEmployee: (id: string) => apiClient.get(`/api/v1/hr/employees/${id}`),
  createEmployee: (data: Record<string, unknown>) => apiClient.post('/api/v1/hr/employees', data),
  getRecruitment: () => apiClient.get('/api/v1/hr/recruitment'),
  getAnalytics: () => apiClient.get('/api/v1/hr/analytics'),
};

export const marketingApi = {
  getCampaigns: (params?: Record<string, unknown>) => apiClient.get('/api/v1/marketing/campaigns', { params }),
  createCampaign: (data: Record<string, unknown>) => apiClient.post('/api/v1/marketing/campaigns', data),
  getAnalytics: () => apiClient.get('/api/v1/marketing/analytics'),
  getLeadSources: () => apiClient.get('/api/v1/marketing/lead-sources'),
};

export const supportApi = {
  getTickets: (params?: Record<string, unknown>) => apiClient.get('/api/v1/support/tickets', { params }),
  getTicket: (id: string) => apiClient.get(`/api/v1/support/tickets/${id}`),
  createTicket: (data: Record<string, unknown>) => apiClient.post('/api/v1/support/tickets', data),
  updateTicket: (id: string, data: Record<string, unknown>) => apiClient.put(`/api/v1/support/tickets/${id}`, data),
  getAnalytics: () => apiClient.get('/api/v1/support/analytics'),
};

export const legalApi = {
  getContracts: (params?: Record<string, unknown>) => apiClient.get('/api/v1/legal/contracts', { params }),
  getContract: (id: string) => apiClient.get(`/api/v1/legal/contracts/${id}`),
  createContract: (data: Record<string, unknown>) => apiClient.post('/api/v1/legal/contracts', data),
  getCompliance: () => apiClient.get('/api/v1/legal/compliance'),
  getRisks: () => apiClient.get('/api/v1/legal/risks'),
};

export const itApi = {
  getInfrastructure: () => apiClient.get('/api/v1/it/infrastructure'),
  getIncidents: (params?: Record<string, unknown>) => apiClient.get('/api/v1/it/incidents', { params }),
  createIncident: (data: Record<string, unknown>) => apiClient.post('/api/v1/it/incidents', data),
  getMetrics: () => apiClient.get('/api/v1/it/metrics'),
  getAlerts: () => apiClient.get('/api/v1/it/alerts'),
};

export const adminApi = {
  getUsers: (params?: Record<string, unknown>) => apiClient.get('/api/v1/admin/users', { params }),
  createUser: (data: Record<string, unknown>) => apiClient.post('/api/v1/admin/users', data),
  updateUser: (id: string, data: Record<string, unknown>) => apiClient.put(`/api/v1/admin/users/${id}`, data),
  deleteUser: (id: string) => apiClient.delete(`/api/v1/admin/users/${id}`),
  getAuditLogs: (params?: Record<string, unknown>) => apiClient.get('/api/v1/admin/audit-logs', { params }),
  getSystemConfig: () => apiClient.get('/api/v1/admin/config'),
};

export const aiApi = {
  chat: (message: string, conversationId?: string) =>
    apiClient.post('/api/v1/ai/chat', { message, conversation_id: conversationId }),
  getConversations: () => apiClient.get('/api/v1/ai/conversations'),
  getWorkflows: () => apiClient.get('/api/v1/ai/workflows'),
  runWorkflow: (workflowId: string, params: Record<string, unknown>) =>
    apiClient.post(`/api/v1/ai/workflows/${workflowId}/run`, params),
  getAgentStatus: () => apiClient.get('/api/v1/ai/agents/status'),
};

export default apiClient;

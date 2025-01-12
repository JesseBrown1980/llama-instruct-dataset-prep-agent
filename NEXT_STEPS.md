# Next Steps for Development

## 1. Multi-Format Input Support
- Add parsers for different data formats:
  - JSON/JSONL structured data
  - CSV/Excel spreadsheets
  - Markdown/Text documents
  - PDF documents with structured data
  - XML/YAML configurations
- Create a unified parser interface:
  ```python
  class DataParser:
      def parse(self, input_path: str) -> KnowledgeBase:
          pass
  ```
- Implement format detection and automatic parser selection

## 2. Flexible Knowledge Base Structure
- Make KB structure configurable via JSON schema:
  ```json
  {
    "structure": {
      "nodes": {
        "type": "object",
        "properties": {
          "id": "string",
          "content": "string",
          "metadata": "object"
        }
      },
      "relationships": {
        "type": "array",
        "items": {
          "source": "string",
          "target": "string",
          "type": "string"
        }
      }
    }
  }
  ```
- Support different KB strategies:
  - Graph-based (current)
  - Document-based (for text chunks)
  - Key-value pairs
  - Hierarchical structures
  - Vector embeddings
- Allow custom indexing and querying methods

## 3. Multiple Dataset Formats
### Dolly Format (Current)
```json
{
  "instruction": "...",
  "context": "...",
  "response": "..."
}
```

### Raw Text Format
```json
{
  "text": "...",
  "metadata": {
    "source": "...",
    "category": "...",
    "tags": []
  }
}
```

- Add format converters
- Support custom output templates
- Enable batch processing modes

## 4. API and Authentication
- Implement FastAPI backend:
  ```python
  from fastapi import FastAPI, Security
  from fastapi.security import OAuth2PasswordBearer
  
  app = FastAPI()
  
  @app.post("/api/v1/dataset")
  async def create_dataset(
      input_file: UploadFile,
      config: DatasetConfig,
      current_user: User = Security(get_current_user)
  ):
      pass
  ```
- Add authentication:
  - OAuth2 with JWT
  - API key support
  - Role-based access control
- Implement rate limiting and quotas

## 5. Configurable Question Generation
- Make prompts configurable:
  ```yaml
  prompts:
    question_generation:
      template: |
        Given the following context, generate {num_questions} questions that:
        {criteria}
        
        Context: {context}
      parameters:
        num_questions: 5
        criteria:
          - "Are specific and focused"
          - "Cover key concepts"
          - "Vary in complexity"
  ```
- Support different question types:
  - Multiple choice
  - Open-ended
  - True/False
  - Fill in the blanks
- Allow custom validation rules

## 6. Containerization and Deployment
### Docker Setup
```dockerfile
FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.9 \
    python3-pip \
    git

# Install Ollama
RUN curl -fsSL https://ollama.ai/install.sh | sh

# Copy application
COPY . /app
WORKDIR /app

# Install dependencies
RUN pip install -r requirements.txt

# Setup volume for data persistence
VOLUME /app/data

# Start services
CMD ["./start.sh"]
```

### Deployment Requirements
- GPU-enabled instance (e.g., AWS g4dn.xlarge)
- Persistent storage for:
  - Knowledge bases
  - Generated datasets
  - Model caches
- Environment variables for configuration
- Health monitoring and logging

### Docker Compose for Development
```yaml
version: '3.8'
services:
  app:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./data:/app/data
    environment:
      - CUDA_VISIBLE_DEVICES=0
      - MAX_MEMORY=16G
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

## 7. Billing and Cost Management

### Cost Calculation System
```python
class CostCalculator:
    def __init__(self):
        self.token_costs = {
            "llama2": 0.0001,  # per token
            "gpt4": 0.03,      # per 1K tokens
        }
        self.compute_costs = {
            "gpu_hour": 0.50,  # per hour
            "storage_gb": 0.05  # per GB per month
        }
    
    def calculate_job_cost(self, job_metrics: JobMetrics) -> Cost:
        return Cost(
            token_cost=self._calc_token_cost(job_metrics.tokens),
            compute_cost=self._calc_compute_cost(job_metrics.duration),
            storage_cost=self._calc_storage_cost(job_metrics.storage_used)
        )
```

### Usage Tracking
```python
@dataclass
class JobMetrics:
    start_time: datetime
    end_time: datetime
    tokens_generated: int
    tokens_processed: int
    storage_used: float  # in GB
    gpu_time: float     # in hours
    
    @property
    def duration(self) -> timedelta:
        return self.end_time - self.start_time
```

### Stripe Integration
```python
from stripe import Customer, Subscription, Product

class BillingManager:
    def __init__(self, stripe_key: str):
        self.stripe = stripe.Stripe(stripe_key)
        
    async def create_subscription(
        self,
        customer: Customer,
        plan: str
    ) -> Subscription:
        return await self.stripe.subscriptions.create(
            customer=customer.id,
            items=[{"price": self.get_plan_price_id(plan)}],
            payment_behavior="default_incomplete",
            expand=["latest_invoice.payment_intent"]
        )
    
    def calculate_usage(self, customer: Customer) -> dict:
        """Calculate usage for current billing period."""
        return {
            "token_usage": self._get_token_usage(customer),
            "compute_time": self._get_compute_time(customer),
            "storage_used": self._get_storage_usage(customer)
        }
```

### Pricing Plans
```json
{
  "plans": {
    "basic": {
      "price_monthly": 49.99,
      "includes": {
        "tokens_per_month": 100000,
        "compute_hours": 10,
        "storage_gb": 5
      },
      "overage_rates": {
        "tokens": 0.0002,
        "compute_hour": 0.75,
        "storage_gb": 0.08
      }
    },
    "pro": {
      "price_monthly": 199.99,
      "includes": {
        "tokens_per_month": 500000,
        "compute_hours": 50,
        "storage_gb": 25
      },
      "overage_rates": {
        "tokens": 0.00015,
        "compute_hour": 0.60,
        "storage_gb": 0.06
      }
    }
  }
}
```

### Database Schema
```sql
CREATE TABLE usage_records (
    id UUID PRIMARY KEY,
    customer_id UUID REFERENCES customers(id),
    timestamp TIMESTAMP WITH TIME ZONE,
    tokens_used INTEGER,
    compute_seconds INTEGER,
    storage_bytes BIGINT,
    cost_usd DECIMAL(10,2),
    billing_period_start DATE,
    billing_period_end DATE
);

CREATE TABLE billing_cycles (
    id UUID PRIMARY KEY,
    customer_id UUID REFERENCES customers(id),
    start_date DATE,
    end_date DATE,
    plan_id UUID REFERENCES plans(id),
    base_amount_usd DECIMAL(10,2),
    overage_amount_usd DECIMAL(10,2),
    status VARCHAR(20)
);
```

### Features
1. **Real-time Cost Tracking**
   - Token usage monitoring
   - Compute time tracking
   - Storage utilization
   - Cost estimation during job execution

2. **Billing Integration**
   - Stripe subscription management
   - Usage-based billing
   - Automatic invoicing
   - Payment processing

3. **Cost Optimization**
   - Usage analytics
   - Cost forecasting
   - Optimization recommendations
   - Budget alerts

4. **Reporting**
   - Detailed usage reports
   - Cost breakdown by component
   - Billing history
   - Usage trends

5. **Admin Dashboard**
   - Customer management
   - Plan configuration
   - Usage monitoring
   - Revenue analytics

### Implementation Steps
1. Set up Stripe integration
2. Implement usage tracking
3. Create billing database schema
4. Develop cost calculation system
5. Build admin interface
6. Add reporting features
7. Implement alerting system

### Technical Requirements
- Stripe API integration
- PostgreSQL for usage tracking
- Redis for real-time metrics
- Background workers for usage aggregation
- Monitoring and alerting system

## 8. RunPod Cost Reconciliation

### RunPod API Integration
```python
import runpod

class RunPodCostTracker:
    def __init__(self, api_key: str):
        self.client = runpod.Client(api_key)
        
    async def get_pod_usage(self, pod_id: str, start_time: datetime, end_time: datetime) -> dict:
        """Get pod usage details from RunPod."""
        usage = await self.client.get_pod_metrics(
            pod_id=pod_id,
            start_time=start_time.isoformat(),
            end_time=end_time.isoformat()
        )
        return {
            "gpu_hours": usage["gpuHours"],
            "cost": usage["totalCost"],
            "pod_type": usage["podType"],
            "gpu_type": usage["gpuType"]
        }

class CostReconciliation:
    def __init__(self, runpod_tracker: RunPodCostTracker, local_calculator: CostCalculator):
        self.runpod = runpod_tracker
        self.calculator = local_calculator
        
    async def reconcile_costs(self, job_id: str) -> CostComparison:
        """Compare RunPod costs with locally calculated costs."""
        # Get job details
        job = await self.get_job_details(job_id)
        
        # Get RunPod costs
        runpod_costs = await self.runpod.get_pod_usage(
            pod_id=job.pod_id,
            start_time=job.start_time,
            end_time=job.end_time
        )
        
        # Get local costs
        local_costs = self.calculator.calculate_job_cost(job.metrics)
        
        # Calculate discrepancy
        discrepancy = {
            "absolute": runpod_costs["cost"] - local_costs.total,
            "percentage": ((runpod_costs["cost"] - local_costs.total) / runpod_costs["cost"]) * 100
        }
        
        return CostComparison(
            job_id=job_id,
            runpod_cost=runpod_costs,
            local_cost=local_costs,
            discrepancy=discrepancy
        )
```

### Cost Comparison Database
```sql
CREATE TABLE cost_comparisons (
    id UUID PRIMARY KEY,
    job_id UUID REFERENCES jobs(id),
    timestamp TIMESTAMP WITH TIME ZONE,
    runpod_cost DECIMAL(10,2),
    local_cost DECIMAL(10,2),
    discrepancy_amount DECIMAL(10,2),
    discrepancy_percentage DECIMAL(5,2),
    reconciled BOOLEAN DEFAULT FALSE,
    notes TEXT
);

CREATE TABLE cost_adjustment_factors (
    id UUID PRIMARY KEY,
    gpu_type VARCHAR(50),
    pod_type VARCHAR(50),
    adjustment_factor DECIMAL(10,4),
    effective_from TIMESTAMP WITH TIME ZONE,
    effective_to TIMESTAMP WITH TIME ZONE,
    last_updated TIMESTAMP WITH TIME ZONE
);
```

### Cost Analysis Dashboard
```python
@app.get("/api/v1/cost-analysis")
async def get_cost_analysis(
    start_date: date,
    end_date: date,
    current_user: User = Security(get_current_user)
) -> CostAnalysis:
    """Get cost analysis and comparison metrics."""
    return {
        "summary": {
            "total_runpod_cost": await get_total_runpod_cost(start_date, end_date),
            "total_local_cost": await get_total_local_cost(start_date, end_date),
            "average_discrepancy": await get_average_discrepancy(start_date, end_date),
            "cost_trend": await get_cost_trend(start_date, end_date)
        },
        "details": {
            "daily_comparison": await get_daily_cost_comparison(start_date, end_date),
            "gpu_type_analysis": await get_gpu_type_cost_analysis(start_date, end_date),
            "job_type_analysis": await get_job_type_cost_analysis(start_date, end_date)
        },
        "recommendations": await generate_cost_recommendations(start_date, end_date)
    }
```

### Reconciliation Features
1. **Automated Cost Tracking**
   - Real-time RunPod cost monitoring
   - Local cost calculation
   - Discrepancy detection
   - Alert system for large variations

2. **Cost Analysis Tools**
   - Historical cost comparison
   - Trend analysis
   - GPU type cost efficiency
   - Job type cost patterns

3. **Adjustment System**
   - Dynamic cost factor adjustment
   - GPU-specific pricing updates
   - Historical adjustment tracking
   - Reconciliation workflow

4. **Reporting**
   - Daily cost reconciliation reports
   - Monthly billing summaries
   - Discrepancy analysis
   - Cost optimization suggestions

### Implementation Steps
1. Integrate RunPod API
2. Set up cost comparison database
3. Implement reconciliation logic
4. Create analysis dashboard
5. Add alerting system
6. Build reporting tools
7. Deploy monitoring system

### Technical Requirements
- RunPod API access
- Time series database for metrics
- Analytics dashboard
- Automated reconciliation jobs
- Monitoring and alerting
- Report generation system

## Implementation Priority
1. Multi-Format Input Support
2. API and Authentication
3. Containerization
4. Configurable Question Generation
5. Flexible KB Structure
6. Multiple Dataset Formats
7. Billing and Cost Management
8. RunPod Cost Reconciliation

Each feature should include:
- Unit tests
- Documentation
- Example configurations
- Performance benchmarks
- Migration guides

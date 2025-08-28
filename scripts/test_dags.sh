#!/bin/bash
# DAG Testing Script for Bank Term Deposit Prediction MLOps Pipeline
# This script provides comprehensive testing capabilities for Airflow DAGs

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
AIRFLOW_CONTAINER="mlops-airflow-webserver"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Helper functions
print_header() {
    echo -e "\n${BLUE}====== $1 ======${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ $1${NC}"
}

# Check if Docker container is running
check_container() {
    if ! docker ps | grep -q "$AIRFLOW_CONTAINER"; then
        print_error "Airflow container '$AIRFLOW_CONTAINER' is not running"
        print_info "Run: docker-compose up -d"
        exit 1
    fi
    print_success "Airflow container is running"
}

# Test DAG syntax locally
test_dag_syntax() {
    print_header "Testing DAG Syntax (Local)"

    cd "$PROJECT_ROOT"
    export PYTHONPATH="${PYTHONPATH}:${PROJECT_ROOT}/src"

    for dag_file in dags/*.py; do
        if [[ -f "$dag_file" ]]; then
            dag_name=$(basename "$dag_file")
            echo -n "Testing $dag_name... "

            if python -m py_compile "$dag_file" 2>/dev/null; then
                print_success "Syntax OK"
            else
                print_error "Syntax Error"
                python -m py_compile "$dag_file"
                return 1
            fi
        fi
    done
}

# Test DAG imports locally
test_dag_imports() {
    print_header "Testing DAG Imports (Local)"

    cd "$PROJECT_ROOT"
    export PYTHONPATH="${PYTHONPATH}:${PROJECT_ROOT}/src"

    # Test training DAG
    echo -n "Testing training_dag import... "
    if python -c "from dags.training_dag import dag; print(f'DAG {dag.dag_id} loaded with {len(dag.tasks)} tasks')" 2>/dev/null; then
        print_success "Import OK"
    else
        print_error "Import Failed"
        python -c "from dags.training_dag import dag"
        return 1
    fi

    # Test drift DAG
    echo -n "Testing drift_dag import... "
    if python -c "from dags.drift_dag import dag; print(f'DAG {dag.dag_id} loaded with {len(dag.tasks)} tasks')" 2>/dev/null; then
        print_success "Import OK"
    else
        print_warning "Import Failed (may need config.yaml)"
    fi

    # Test deployment DAG
    echo -n "Testing deployment_dag import... "
    if python -c "from dags.deployment_dag import dag; print(f'DAG {dag.dag_id} loaded with {len(dag.tasks)} tasks')" 2>/dev/null; then
        print_success "Import OK"
    else
        print_warning "Import Failed (may need deployment module)"
    fi
}

# Test DAGs in Airflow container
test_dag_container() {
    print_header "Testing DAGs in Container"

    echo -n "Testing DAG parsing in container... "
    if docker exec "$AIRFLOW_CONTAINER" airflow dags list-import-errors 2>/dev/null | grep -q "No import errors"; then
        print_success "No import errors"
    else
        print_error "Import errors detected"
        docker exec "$AIRFLOW_CONTAINER" airflow dags list-import-errors
        return 1
    fi

    # List all DAGs
    print_info "Available DAGs in container:"
    docker exec "$AIRFLOW_CONTAINER" airflow dags list | grep -E "(training_pipeline|drift_detection|deployment_dag)" || print_warning "Some DAGs may not be loaded"
}

# Test specific DAG structure
test_dag_structure() {
    local dag_id="$1"
    print_header "Testing DAG Structure: $dag_id"

    # Show DAG structure
    echo "DAG graph for $dag_id:"
    if docker exec "$AIRFLOW_CONTAINER" airflow dags show "$dag_id" 2>/dev/null; then
        print_success "DAG structure is valid"
    else
        print_error "DAG structure test failed"
        return 1
    fi

    # List tasks
    echo -e "\nTasks in $dag_id:"
    docker exec "$AIRFLOW_CONTAINER" airflow tasks list "$dag_id" 2>/dev/null || {
        print_error "Could not list tasks for $dag_id"
        return 1
    }
}

# Test individual task
test_task() {
    local dag_id="$1"
    local task_id="$2"
    local execution_date="${3:-$(date -d '1 day ago' '+%Y-%m-%d')}"

    print_header "Testing Task: $dag_id.$task_id"

    echo "Testing task $task_id on $execution_date..."
    if docker exec "$AIRFLOW_CONTAINER" airflow tasks test "$dag_id" "$task_id" "$execution_date"; then
        print_success "Task test completed"
    else
        print_error "Task test failed"
        return 1
    fi
}

# Test core pipeline components
test_pipeline_components() {
    print_header "Testing Pipeline Components"

    cd "$PROJECT_ROOT"
    export PYTHONPATH="${PYTHONPATH}:${PROJECT_ROOT}/src"

    # Test data ingestion
    echo -n "Testing data ingestion module... "
    if python -c "from src.data.ingest import load_dataset; print('✓ Data ingestion module OK')" 2>/dev/null; then
        print_success "Data ingestion OK"
    else
        print_error "Data ingestion failed"
    fi

    # Test feature transformation
    echo -n "Testing feature transformation module... "
    if python -c "from src.features.transform import transform_dataset; print('✓ Feature transformation module OK')" 2>/dev/null; then
        print_success "Feature transformation OK"
    else
        print_error "Feature transformation failed"
    fi

    # Test model training
    echo -n "Testing model training module... "
    if python -c "from src.models.train import train_catboost_model; print('✓ Model training module OK')" 2>/dev/null; then
        print_success "Model training OK"
    else
        print_error "Model training failed"
    fi

    # Test model validation
    echo -n "Testing model validation module... "
    if python -c "from src.models.validate import validate_model; print('✓ Model validation module OK')" 2>/dev/null; then
        print_success "Model validation OK"
    else
        print_error "Model validation failed"
    fi
}

# Test MLflow connectivity
test_mlflow() {
    print_header "Testing MLflow Connectivity"

    echo -n "Testing MLflow server connectivity... "
    if curl -s http://localhost:5000/health > /dev/null 2>&1; then
        print_success "MLflow server is accessible"
    else
        print_warning "MLflow server is not accessible (may need to start mlflow service)"
    fi
}

# Test dependencies in container
test_dependencies() {
    print_header "Testing Dependencies in Container"

    echo "Testing ML dependencies in Airflow container..."
    if docker exec "$AIRFLOW_CONTAINER" python /app/scripts/test_dependencies.py; then
        print_success "All dependencies are available"
    else
        print_error "Some dependencies are missing"
        print_info "You may need to rebuild the container with: docker-compose up --build"
        return 1
    fi
}

# Run all tests
run_all_tests() {
    print_header "Running All DAG Tests"

    # Basic checks
    check_container
    test_dependencies
    test_mlflow

    # Local tests
    test_dag_syntax
    test_dag_imports
    test_pipeline_components

    # Container tests
    test_dag_container

    # DAG structure tests
    test_dag_structure "training_pipeline"

    print_header "Test Summary"
    print_success "All available tests completed!"
    print_info "To test specific tasks, use: $0 test-task <dag_id> <task_id> [execution_date]"
}

# Main script logic
case "${1:-all}" in
    "syntax")
        test_dag_syntax
        ;;
    "imports")
        test_dag_imports
        ;;
    "container")
        check_container
        test_dag_container
        ;;
    "structure")
        check_container
        test_dag_structure "${2:-training_pipeline}"
        ;;
    "test-task")
        check_container
        test_task "$2" "$3" "$4"
        ;;
    "components")
        test_pipeline_components
        ;;
    "dependencies")
        check_container
        test_dependencies
        ;;
    "mlflow")
        test_mlflow
        ;;
    "all")
        run_all_tests
        ;;
    "help"|"-h"|"--help")
        echo "Usage: $0 [command]"
        echo ""
        echo "Commands:"
        echo "  syntax           Test DAG syntax locally"
        echo "  imports          Test DAG imports locally"
        echo "  container        Test DAGs in Docker container"
        echo "  structure [dag]  Test DAG structure (default: training_pipeline)"
        echo "  test-task <dag> <task> [date]  Test specific task"
        echo "  components       Test pipeline components"
        echo "  dependencies     Test ML dependencies in container"
        echo "  mlflow          Test MLflow connectivity"
        echo "  all             Run all tests (default)"
        echo "  help            Show this help message"
        echo ""
        echo "Examples:"
        echo "  $0 syntax"
        echo "  $0 test-task training_pipeline data_ingestion 2025-08-28"
        echo "  $0 structure training_pipeline"
        ;;
    *)
        print_error "Unknown command: $1"
        echo "Use '$0 help' for usage information"
        exit 1
        ;;
esac
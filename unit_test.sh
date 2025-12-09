#!/bin/bash

# Default values
SKIP_SLOW=true
EXIT_ON_FAIL=true
SHOW_OUTPUT=false
VERBOSE=false
SPECIFIC_PATH="aria/tests"
SPECIFIC_TEST=""
COVERAGE=false
PARALLEL=false

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Determine Python command
if command -v python3 >/dev/null 2>&1; then
    PYTHON_CMD="python3"
elif command -v python >/dev/null 2>&1; then
    PYTHON_CMD="python"
else
    echo -e "${RED}Error: Neither python3 nor python command found${NC}"
    exit 1
fi

# Help message
show_help() {
    echo -e "${BLUE}Usage: $0 [options]${NC}"
    echo
    echo "Options:"
    echo "  -h, --help           Show this help message"
    echo "  -a, --all           Include slow tests"
    echo "  -c, --continue      Continue on test failure"
    echo "  -p, --print         Show test output (print statements)"
    echo "  -v, --verbose       Show verbose output"
    echo "  -t, --test PATH     Run specific test file or directory"
    echo "  -n, --name PATTERN  Run tests matching pattern"
    echo "  --coverage          Generate coverage report"
    echo "  --parallel          Run tests in parallel"
    echo
    echo "Examples:"
    echo "  $0                   # Run all tests except slow ones"
    echo "  $0 -a -p            # Run all tests including slow ones with output"
    echo "  $0 -t path/to/test  # Run specific test file or directory"
    echo "  $0 -n test_name     # Run tests matching pattern"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        -a|--all)
            SKIP_SLOW=false
            shift
            ;;
        -c|--continue)
            EXIT_ON_FAIL=false
            shift
            ;;
        -p|--print)
            SHOW_OUTPUT=true
            shift
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        -t|--test)
            SPECIFIC_PATH="$2"
            shift 2
            ;;
        -n|--name)
            SPECIFIC_TEST="$2"
            shift 2
            ;;
        --coverage)
            COVERAGE=true
            shift
            ;;
        --parallel)
            PARALLEL=true
            shift
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            show_help
            exit 1
            ;;
    esac
done

# Build pytest command
PYTEST_CMD="$PYTHON_CMD -m pytest -W ignore::RuntimeWarning:unittest.case"

# Add options based on flags
[[ "$SKIP_SLOW" = true ]] && PYTEST_CMD+=" -m 'not slow'"
[[ "$EXIT_ON_FAIL" = true ]] && PYTEST_CMD+=" -x"
[[ "$SHOW_OUTPUT" = true ]] && PYTEST_CMD+=" -rP"
[[ "$VERBOSE" = true ]] && PYTEST_CMD+=" -v"
[[ -n "$SPECIFIC_TEST" ]] && PYTEST_CMD+=" -k $SPECIFIC_TEST"
[[ "$COVERAGE" = true ]] && PYTEST_CMD+=" --cov=aria --cov-report=term-missing"
[[ "$PARALLEL" = true ]] && PYTEST_CMD+=" -n auto"

# Add the test path
PYTEST_CMD+=" $SPECIFIC_PATH"

# Print configuration
echo -e "${YELLOW}Test Configuration:${NC}"
echo -e "  Skip slow tests: ${GREEN}$SKIP_SLOW${NC}"
echo -e "  Exit on failure: ${GREEN}$EXIT_ON_FAIL${NC}"
echo -e "  Show test output: ${GREEN}$SHOW_OUTPUT${NC}"
echo -e "  Verbose mode: ${GREEN}$VERBOSE${NC}"
echo -e "  Coverage report: ${GREEN}$COVERAGE${NC}"
echo -e "  Parallel execution: ${GREEN}$PARALLEL${NC}"
echo -e "  Test path: ${GREEN}$SPECIFIC_PATH${NC}"
[[ -n "$SPECIFIC_TEST" ]] && echo -e "  Test pattern: ${GREEN}$SPECIFIC_TEST${NC}"
echo
echo -e "${YELLOW}Running command:${NC}"
echo -e "${BLUE}$PYTEST_CMD${NC}"
echo

# Run the tests
eval "$PYTEST_CMD"
TEST_EXIT_CODE=$?

# Display summary
echo
if [ $TEST_EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}All tests passed successfully!${NC}"
else
    echo -e "${RED}Tests failed with exit code $TEST_EXIT_CODE${NC}"
fi

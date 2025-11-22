#!/usr/bin/env python3
"""Test script for the enhanced SMT server with advanced arlib features."""

import os
import time
import sys
import subprocess
import atexit

INPUT_PIPE = "/tmp/smt_input"
OUTPUT_PIPE = "/tmp/smt_output"
server_process = None


def start_server():
    """Start the SMT server if it's not already running."""
    global server_process

    # Always clean up any existing pipes to avoid conflicts
    if os.path.exists(INPUT_PIPE):
        os.unlink(INPUT_PIPE)
    if os.path.exists(OUTPUT_PIPE):
        os.unlink(OUTPUT_PIPE)

    print("Starting SMT server...")
    try:
        server_process = subprocess.Popen([sys.executable, "-m", "arlib.cli.smt_server"],
                                        stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        # Wait for pipes to be created
        time.sleep(1.0)  # Give server time to start and create pipes
        for _ in range(50):  # 5 second timeout
            if os.path.exists(INPUT_PIPE) and os.path.exists(OUTPUT_PIPE):
                time.sleep(0.5)  # Give server time to initialize
                print("SMT server started successfully.")
                return
            time.sleep(0.1)
        raise TimeoutError("Timed out waiting for SMT server to start")
    except Exception as e:
        print(f"Failed to start SMT server: {e}")
        sys.exit(1)


def cleanup():
    """Clean up resources when the script exits."""
    global server_process
    if server_process:
        print("Shutting down SMT server...")
        try:
            send_command("exit")
            time.sleep(0.5)
            if server_process.poll() is None:
                server_process.terminate()
                server_process.wait(timeout=2)
            print("SMT server shut down.")
        except Exception as e:
            print(f"Error shutting down server: {e}")
            if server_process.poll() is None:
                server_process.kill()


def send_command(command):
    """Send a command to the SMT server and return the response."""
    try:
        with open(INPUT_PIPE, 'w') as f:
            f.write(command + '\n')
            f.flush()

        time.sleep(0.5)  # Wait for server to process

        with open(OUTPUT_PIPE, 'r') as f:
            for _ in range(200):  # 2 second timeout
                line = f.readline()
                if line:
                    return line.strip()
                time.sleep(0.01)
        return "timeout"
    except Exception as e:
        print(f"Error sending command: {e}")
        return None


def test_basic_functionality():
    """Test basic SMT-LIB2 functionality."""
    print("\n=== Testing Basic Functionality ===")

    # Setup
    for cmd in ["declare-const x Int", "declare-const y Int", "declare-const p Bool"]:
        send_command(cmd)

    for cmd in ["assert (> x 0)", "assert (< y 10)", "assert (= (+ x y) 5)"]:
        send_command(cmd)

    # Test
    result = send_command("check-sat")
    print(f"Result: {result}")
    print(f"Model: {send_command('get-model')}")
    print(f"Values: {send_command('get-value x y')}")


def test_advanced_features():
    """Test advanced arlib features."""
    print("\n=== Testing Advanced Features ===")

    # Simple AllSMT test
    send_command("push")
    send_command("pop")
    send_command("declare-const a Bool")
    send_command("declare-const b Bool")
    send_command("assert (or a b)")
    print(f"AllSMT: {send_command('allsmt a b')}")

    # Simple UNSAT core test
    send_command("push")
    send_command("pop")
    send_command("declare-const p Bool")
    send_command("declare-const q Bool")
    send_command("assert p")
    send_command("assert (not p)")
    print(f"UNSAT result: {send_command('check-sat')}")
    print(f"UNSAT core: {send_command('unsat-core')}")

    # Simple model counting test
    send_command("push")
    send_command("pop")
    send_command("declare-const x Bool")
    send_command("declare-const y Bool")
    send_command("assert (or x y)")
    print(f"Model count: {send_command('count-models')}")


def test_help():
    """Test help command."""
    print("\n=== Testing Help Command ===")
    print(f"Help:\n{send_command('help')}")


def main():
    """Run all tests."""
    atexit.register(cleanup)
    start_server()

    test_basic_functionality()
    test_advanced_features()
    test_help()

    print("\nAll tests completed.")


if __name__ == "__main__":
    main()

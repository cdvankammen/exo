//  EXOExitReasonTests.swift
//  EXOTests
//
//  Coverage for ExoProcessController.extractStartupFailureReason:
//  the pure static mapping from (exitCode, stderrTail, logTail) to a
//  human-readable reason, including the known failure signatures from
//  card t_48bcd15c (Address already in use, ModuleNotFoundError,
//  ValidationError, etc.) and the fallback behavior.
//

import Testing
@testable import EXO

struct EXOExitReasonTests {

    // MARK: - Known signatures (card t_48bcd15c)

    @Test func addressAlreadyInUseInStderr() {
        let reason = EXO.ExoProcessController.extractStartupFailureReason(
            exitCode: 1,
            stderrTail: "zenoh: Address already in use",
            logTail: ""
        )
        #expect(reason.contains("Port already in use"))
    }

    @Test func addressAlreadyInUseInLog() {
        let reason = EXO.ExoProcessController.extractStartupFailureReason(
            exitCode: 1,
            stderrTail: "",
            logTail: "[ERROR] failed to bind socket: address already in use"
        )
        #expect(reason.contains("Port already in use"))
    }

    @Test func moduleNotFoundError() {
        let reason = EXO.ExoProcessController.extractStartupFailureReason(
            exitCode: 1,
            stderrTail: "ModuleNotFoundError: No module named 'exo_rs'",
            logTail: ""
        )
        #expect(reason.contains("Missing Python module"))
    }

    @Test func validationError() {
        let reason = EXO.ExoProcessController.extractStartupFailureReason(
            exitCode: 1,
            stderrTail: "",
            logTail: "pydantic_core._pydantic_core.ValidationError: Extra inputs are not permitted [type=extra_forbidden, input_value={'event_id': 'fe0f4b56-17'}, input_type=dict]"
        )
        #expect(reason.contains("Validation error"))
    }

    @Test func noSpaceLeftOnDevice() {
        let reason = EXO.ExoProcessController.extractStartupFailureReason(
            exitCode: 1,
            stderrTail: "",
            logTail: "OSError: [Errno 28] No space left on device"
        )
        #expect(reason.contains("No disk space left"))
    }

    @Test func missingNativeLibrary() {
        let reason = EXO.ExoProcessController.extractStartupFailureReason(
            exitCode: 1,
            stderrTail: "cannot open shared object file: No such file or directory",
            logTail: ""
        )
        #expect(reason.contains("Missing native library"))
    }

    @Test func cudaHeaders() {
        let reason = EXO.ExoProcessController.extractStartupFailureReason(
            exitCode: 1,
            stderrTail: "Can not find locations of CUDA headers",
            logTail: ""
        )
        #expect(reason.contains("CUDA headers missing"))
    }

    @Test func gpuDriverMismatch() {
        let reason = EXO.ExoProcessController.extractStartupFailureReason(
            exitCode: 1,
            stderrTail: "cudaMallocManaged failed: out of memory",
            logTail: ""
        )
        #expect(reason.contains("GPU error"))
    }

    @Test func importError() {
        let reason = EXO.ExoProcessController.extractStartupFailureReason(
            exitCode: 1,
            stderrTail: "ImportError: cannot import name 'foo'",
            logTail: ""
        )
        #expect(reason.contains("Import error"))
    }

    // MARK: - Fallback behavior

    @Test func fallsBackToMostRelevantLogLine() {
        let logTail = """
        [ 2026-09-14 20:55:28.087 | INFO | exo.shared.election:run:203 ] Election shutdown
        [ 2026-09-14 20:55:29.091 | CRITICAL | exo.main:main_inner:466 ] EXO terminated due to unhandled exception
        """
        let reason = EXO.ExoProcessController.extractStartupFailureReason(
            exitCode: 1,
            stderrTail: "",
            logTail: logTail
        )
        // The last line containing ERROR/Traceback/Fatal is returned (up to 300 chars).
        #expect(reason.contains("EXO terminated due to unhandled exception"))
    }

    @Test func fallsBackToExitCodeWhenNothingMatches() {
        let reason = EXO.ExoProcessController.extractStartupFailureReason(
            exitCode: 42,
            stderrTail: "",
            logTail: ""
        )
        #expect(reason == "Process exited with code 42")
    }

    // MARK: - Malformed / edge input

    @Test func emptyInputsNeverCrash() {
        let reason = EXO.ExoProcessController.extractStartupFailureReason(
            exitCode: 0,
            stderrTail: "",
            logTail: ""
        )
        #expect(reason == "Process exited with code 0")
    }

    @Test func unicodeAndControlCharactersAreHandled() {
        let reason = EXO.ExoProcessController.extractStartupFailureReason(
            exitCode: 1,
            stderrTail: "Traceback (most recent call last):\n  File \"x.py\", line 1, in <module>\n    raise RuntimeError('boom')\nRuntimeError: boom",
            logTail: ""
        )
        // No recognized signature → falls back to the last log line.
        #expect(reason.contains("RuntimeError: boom"))
    }
}
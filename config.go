package main

import "log"

const (
	Version         = "0.20.0"
	DefaultPort     = 11434
	DefaultHost     = "0.0.0.0"
	DefaultLlamaURL = "http://127.0.0.1:8080"
)

var verboseLevel int // 0=silent, 1=verbose, 2=very verbose

func logVerbose(format string, args ...any) {
	if verboseLevel >= 1 {
		log.Printf(format, args...)
	}
}

func logVeryVerbose(format string, args ...any) {
	if verboseLevel >= 2 {
		log.Printf(format, args...)
	}
}

func truncateString(s string, maxLen int) string {
	if len(s) <= maxLen {
		return s
	}
	return s[:maxLen] + "..."
}

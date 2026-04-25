package main

import (
	"log"
	"os"
)

const (
	Version     = "0.1.0"
	DefaultPort = 11434
)

var verboseLevel int // 0=silent, 1=verbose, 2=very verbose

func getEnv(key, defaultVal string) string {
	if val := os.Getenv(key); val != "" {
		return val
	}
	return defaultVal
}

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

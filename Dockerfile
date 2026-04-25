FROM golang:1.26-bookworm AS builder
WORKDIR /src
COPY . .
RUN CGO_ENABLED=0 GOOS=linux go build -ldflags="-s -w" -o /o_llama_relay .

FROM alpine:latest
COPY --from=builder /o_llama_relay /o_llama_relay
EXPOSE 11434
ENTRYPOINT ["/o_llama_relay"]

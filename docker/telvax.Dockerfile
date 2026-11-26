# syntax=docker/dockerfile:1
# The Telvax version of docker/Dockerfile: the model is trained while the image is built (as `RUN python src/main.py`),
# then the service is one binary in a distroless image, with the model next to it.
#   docker build -f docker/telvax.Dockerfile -t sales-prediction-telvax .
# Started from `telvax build --serve --dockerfile`, which writes a Dockerfile for a program without dependencies; this one
# installs the packages of telvax/telvax.toml first.

ARG TELVAX_IMAGE=ghcr.io/telvax/telvax:latest
FROM ${TELVAX_IMAGE} AS builder
WORKDIR /app/telvax
COPY telvax/ ./
COPY data/sales_data.csv /app/data/sales_data.csv
RUN tvpm install
RUN tvpm run main --model-path /out/models/sales_predictor.tvxmodel
RUN telvax build --serve src/service.tvx -o /out/service

FROM gcr.io/distroless/cc-debian12:nonroot
COPY --from=builder /out/service /service
COPY --from=builder /out/models /models
ENV PORT=8000 MODEL_PATH=/models/sales_predictor.tvxmodel
EXPOSE 8000
HEALTHCHECK --interval=10s --timeout=4s --start-period=10s CMD ["/service", "--healthcheck"]
USER 65532:65532
ENTRYPOINT ["/service"]

terraform {
  backend "gcs" {
    bucket = "mlb-iris-production-terraform-state"
    prefix = "mlb-iris-lang-graph/dev"
  }
}

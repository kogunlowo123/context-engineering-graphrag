variable "name" {
  description = "Resource name prefix"
  type        = string
  default     = "context-engineering-graphrag"
}

variable "region" {
  description = "AWS region"
  type        = string
}

variable "vpc_id" {
  description = "VPC hosting the EKS cluster"
  type        = string
}

variable "private_subnet_ids" {
  description = "Private subnets for the database (at least two AZs)"
  type        = list(string)
  validation {
    condition     = length(var.private_subnet_ids) >= 2
    error_message = "Provide subnets in at least two availability zones."
  }
}

variable "cluster_security_group_id" {
  description = "Security group of the EKS nodes allowed to reach PostgreSQL"
  type        = string
}

variable "eks_oidc_provider_arn" {
  description = "ARN of the EKS cluster IAM OIDC provider"
  type        = string
}

variable "eks_oidc_issuer_url" {
  description = "EKS cluster OIDC issuer URL"
  type        = string
}

variable "namespace" {
  description = "Kubernetes namespace of the platform"
  type        = string
  default     = "context-engineering"
}

variable "service_account" {
  description = "Kubernetes service account of the platform"
  type        = string
  default     = "context-engineering-graphrag"
}

variable "db_instance_class" {
  description = "RDS instance class"
  type        = string
  default     = "db.r6g.xlarge"
}

variable "db_allocated_storage_gb" {
  description = "Initial database storage"
  type        = number
  default     = 50
}

variable "bedrock_foundation_model_ids" {
  description = "Bedrock foundation models the platform may invoke (embeddings, extraction, summaries)"
  type        = list(string)
  default     = ["amazon.titan-embed-text-v2:0", "amazon.nova-pro-v1:0"]
}

variable "bedrock_inference_profile_arns" {
  description = "Inference profile ARNs (e.g. Claude cross-region profiles) the platform may invoke"
  type        = list(string)
  default     = []
}

variable "audit_retention_days" {
  description = "Object Lock compliance retention for audit chains"
  type        = number
  default     = 2555 # 7 years
}

variable "tags" {
  description = "Tags applied to every resource"
  type        = map(string)
  default     = { "app.kubernetes.io/part-of" = "context-engineering-graphrag" }
}

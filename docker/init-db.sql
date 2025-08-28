-- Database initialization script for MLflow and Airflow
-- This script creates separate databases and users for MLflow and Airflow services

-- Create MLflow user first
CREATE USER mlflow WITH ENCRYPTED PASSWORD 'mlflow';
CREATE USER airflow WITH ENCRYPTED PASSWORD 'airflow';

-- Create databases with the users as owners
CREATE DATABASE mlflow OWNER mlflow;
CREATE DATABASE airflow OWNER airflow;

-- Grant additional permissions
ALTER USER mlflow CREATEDB;
ALTER USER airflow CREATEDB;
---
id: POL-02
title: Passwords and MFA
type: policy
date: '2025-08-01'
section_slugs:
  Password Requirements: requisitos-de-contrasena
  Second Factor: segundo-factor
  Service Accounts: cuentas-de-servicio
---

# POL-02 · Passwords and MFA

## Password Requirements

Minimum 14 characters, no periodic expiration unless suspected compromise. The account is locked for 15 minutes after 10 failed attempts.

## Second Factor

MFA is required for email, VPN, ERP Orion, and administration consoles. Authenticator app or physical key are accepted; SMS is not permitted.

## Service Accounts

Service accounts use 32-character passwords stored in the secret vault and are rotated every 180 days.

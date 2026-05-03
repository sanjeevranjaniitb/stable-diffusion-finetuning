#!/bin/bash
set -e

# Fix macOS SSL certs for conda environments
CERT_FILE="/tmp/macos_certs.pem"
/usr/bin/security find-certificate -a -p /System/Library/Keychains/SystemRootCertificates.keychain > "$CERT_FILE" 2>/dev/null
/usr/bin/security find-certificate -a -p /Library/Keychains/System.keychain >> "$CERT_FILE" 2>/dev/null

export SSL_CERT_FILE="$CERT_FILE"
export REQUESTS_CA_BUNDLE="$CERT_FILE"

conda run -n sd-inpaint-finetune python run_demo.py

"""Mitigation advice shown after an attack is detected.

Keys are matched case-insensitively against the predicted class name; unknown
classes fall back to a generic playbook. Works for CICIDS-style classes and for
NSL-KDD style names (Probe, R2L, U2R) too."""

_PLAYBOOKS = {
    "dos": {
        "title": "Denial of Service (DoS)", "severity": "High", "icon": "🔥",
        "steps": ["Block or rate-limit the suspicious source IPs at the firewall",
                  "Enable SYN-cookie / connection-rate limits on the server",
                  "Monitor bandwidth and open-connection counts for sudden spikes",
                  "Check web-server logs for abnormal or repeated requests (e.g. slow HTTP)",
                  "Put a reverse proxy / load balancer in front of the exposed service"]},
    "ddos": {
        "title": "Distributed Denial of Service (DDoS)", "severity": "Critical", "icon": "🌊",
        "steps": ["Activate upstream DDoS protection (CDN / scrubbing service / ISP null-routing)",
                  "Enable rate limiting and geo/IP reputation filtering",
                  "Scale out or fail over to unaffected servers while the attack lasts",
                  "Capture traffic samples to identify the common attack signature",
                  "Notify the ISP / hosting provider and open an incident ticket"]},
    "portscan": {
        "title": "Port Scan (reconnaissance)", "severity": "Medium", "icon": "🔎",
        "steps": ["Block the scanning IP and add it to a watch-list",
                  "Close or firewall every port that does not need to be public",
                  "Enable IDS/IPS scan-detection rules and alerting",
                  "Review which services responded - this is what the attacker now knows",
                  "Expect a follow-up attack on any open service and patch it now"]},
    "bruteforce": {
        "title": "Brute-force login attempt", "severity": "High", "icon": "🔑",
        "steps": ["Lock or throttle the targeted accounts after repeated failures",
                  "Block the offending IPs (fail2ban or equivalent)",
                  "Enforce strong passwords and multi-factor authentication",
                  "Move SSH/FTP off default ports or restrict them to a VPN / allow-list",
                  "Audit authentication logs for any successful login from the same source"]},
    "webattack": {
        "title": "Web application attack", "severity": "High", "icon": "🕸️",
        "steps": ["Enable or tighten Web Application Firewall (WAF) rules",
                  "Validate and sanitise all user input (SQL injection / XSS protection)",
                  "Review web-server and database logs for the offending requests",
                  "Patch the web framework and plugins; rotate any exposed credentials",
                  "Run an authenticated vulnerability scan against the application"]},
    "bot": {
        "title": "Botnet activity", "severity": "Critical", "icon": "🤖",
        "steps": ["Isolate the infected host from the network immediately",
                  "Block the command-and-control IPs/domains at the firewall and DNS",
                  "Run a full malware scan and reimage the host if needed",
                  "Reset credentials used on that host",
                  "Hunt for the same indicators on other machines"]},
    "probe": {
        "title": "Probe / reconnaissance", "severity": "Medium", "icon": "🔎",
        "steps": ["Block the scanning source", "Close unused ports and services",
                  "Enable IDS scan-detection rules", "Review exposed services"]},
    "r2l": {
        "title": "Remote-to-Local (R2L) access attempt", "severity": "High", "icon": "🚪",
        "steps": ["Block the source IP", "Enforce MFA on remote services",
                  "Audit accounts for unauthorised access", "Patch exposed services"]},
    "u2r": {
        "title": "User-to-Root privilege escalation", "severity": "Critical", "icon": "⬆️",
        "steps": ["Isolate the affected host", "Revoke elevated sessions and rotate credentials",
                  "Patch the exploited vulnerability", "Review sudo / admin logs"]},
}

_GENERIC = {"title": "Suspicious network activity", "severity": "Medium", "icon": "⚠️",
            "steps": ["Isolate or closely monitor the affected host",
                      "Block the suspicious source IP while investigating",
                      "Review firewall and server logs around the flagged time",
                      "Escalate to the security team and record the incident"]}

_NORMAL = {"title": "Normal traffic", "severity": "None", "icon": "✅",
           "steps": ["No action needed - keep routine monitoring enabled"]}


def get_recommendations(predicted_class: str, normal_class: str = "Normal") -> dict:
    """Return {'title', 'severity', 'icon', 'steps'} for a predicted class."""
    if str(predicted_class).lower() == normal_class.lower():
        return _NORMAL
    return _PLAYBOOKS.get(str(predicted_class).lower().replace(" ", "").replace("-", ""), _GENERIC)

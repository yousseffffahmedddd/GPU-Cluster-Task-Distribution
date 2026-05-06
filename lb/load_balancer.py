"""Compatibility wrapper for the external NGINX-style load balancer.

The project originally had an in-process LoadBalancer class. The architecture is
now corrected: load balancing is represented as an external reverse proxy layer.

Keep this file so imports from older code still work, but the real implementation
is `NginxReverseProxy` in `lb/nginx_proxy.py`.
"""

from lb.nginx_proxy import NginxReverseProxy


# Backward-compatible name used by main.py and older project files.
LoadBalancer = NginxReverseProxy

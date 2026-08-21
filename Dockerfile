FROM python:3.14-alpine AS executor

WORKDIR /app

RUN apk --no-cache add \
      shadow \
      su-exec \
      uv \
      bash
RUN useradd -m admin

COPY requirements.txt /tmp/requirements.txt
RUN uv pip install --system -r /tmp/requirements.txt

# Run as a user whose uid/gid match the bind-mounted /app owner, so files
# created inside the container aren't root-owned on the host. `su-exec`
# passes the argv through verbatim — no string re-splitting, so quoted
# arguments (e.g. `python -c "print(1)"`) survive intact.
ENTRYPOINT ["sh", "-c", "\
    if [ \"$0\" = sh ] && [ $# -eq 0 ]; then \
        set -- bash; \
    else \
        set -- \"$0\" \"$@\"; \
    fi; \
    if [ $(stat -c '%u' /app) -eq 0 ]; then \
        exec \"$@\"; \
    else \
        groupmod -g $(stat -c '%u' /app) admin; \
        usermod -u $(stat -c '%u' /app) -g $(stat -c '%u' /app) admin; \
        ln -sf /app/.bash_history /home/admin/.bash_history; \
        chown admin:admin /home/admin; \
        export HOME=/home/admin; \
        exec su-exec admin \"$@\"; \
    fi \
    "]
CMD []

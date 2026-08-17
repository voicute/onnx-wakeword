FROM voicute/voicute-wyoming:1.0.0

# Add-on entrypoint: reads Home Assistant options from /data/options.json and
# launches the Wyoming service. Plain Python (no bash/bashio) — the base image
# is python:3.10-slim, which has neither.
COPY entrypoint.py /entrypoint.py
RUN chmod a+x /entrypoint.py

# Override the base image's ENTRYPOINT (python wyoming/wyoming_voicute.py),
# otherwise CMD below would be appended to it as an argument.
ENTRYPOINT ["python", "/entrypoint.py"]
CMD []

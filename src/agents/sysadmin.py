"""Sysadmin — general system assistant and troubleshooter."""

from src.agent_base import AgentBase


class SysadminAgent(AgentBase):
    agent_role = "sysadmin"
    system_prompt_path = "sysadmin.txt"

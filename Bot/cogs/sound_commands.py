import discord
from discord.ext import commands


class SoundCommands(commands.Cog, name="SoundCommands"):
    def __init__(self, bot):
        self.bot = bot

    # Your commands go here


def setup(bot):
    bot.add_cog(SoundCommands(bot))
import discord
from discord.ext import commands


class UtilCommands(commands.Cog, name="UtilCommands"):
    def __init__(self, bot):
        self.bot = bot

    # Your commands go here


def setup(bot):
    bot.add_cog(UtilCommands(bot))
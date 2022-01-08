import discord
from discord.ext import commands


class ErrorHandling(commands.Cog, name="errorhandling"):
    def __init__(self, bot):
        self.bot = bot

    # Your commands go here


def setup(bot):
    bot.add_cog(ErrorHandling(bot))
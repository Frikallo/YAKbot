import discord
from discord.ext import commands


class Help(commands.Cog, name="Help"):
    def __init__(self, bot):
        self.bot = bot

    # Your commands go here


def setup(bot):
    bot.add_cog(Help(bot))
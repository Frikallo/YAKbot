import discord
from discord.ext import commands


class NaturalLanguageCommands(commands.Cog, name="NaturalLanguageCommands"):
    def __init__(self, bot):
        self.bot = bot

    # Your commands go here


def setup(bot):
    bot.add_cog(NaturalLanguageCommands(bot))
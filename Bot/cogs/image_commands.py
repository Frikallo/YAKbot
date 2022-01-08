import discord
from discord.ext import commands


class ImageCommands(commands.Cog, name="imagecommands"):
    def __init__(self, bot):
        self.bot = bot

    # Your commands go here


def setup(bot):
    bot.add_cog(ImageCommands(bot))
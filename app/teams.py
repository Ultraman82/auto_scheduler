import pymsteams


def send_edgar(writer, comment):
    url_hooks = "https://thkglobal.webhook.office.com/webhookb2/0d7263c1-37f8-4ee4-be65-a7cfb332291a@df8ccede-71dd-4444-8311-2e913ebfa1fa/IncomingWebhook/e906e8996ea241f6b5d649df0f862478/37695268-c0cb-4348-ab67-ce782cd1215b"
    myTeamsMessage = pymsteams.connectorcard(url_hooks)
    myTeamsMessage.title(writer)
    myTeamsMessage.text(comment)
    print(myTeamsMessage.send())


# myTeamsMessage.summary("Test Message")

# myTeamsPotentialAction1 = pymsteams.potentialaction(_name="Submit")
# myTeamsPotentialAction1.addInput(
#     "TextInput", "comment", "Reason for unrealse", False)
# myTeamsPotentialAction1.addAction(
#     "HttpPost", "Add Comment", "https://dev-dx.thkma.com/message-app/teams")

# myTeamsPotentialAction2 = pymsteams.potentialaction(_name = "Set due date")
# myTeamsPotentialAction2.addInput("DateInput","dueDate","Enter due date")
# myTeamsPotentialAction2.addAction("HttpPost","save","https://...")

# myTeamsPotentialAction3 = pymsteams.potentialaction(_name = "Change Status")
# myTeamsPotentialAction3.choices.addChoices("In progress","0")
# myTeamsPotentialAction3.choices.addChoices("Active","1")
# myTeamsPotentialAction3.addInput("MultichoiceInput","list","Select a status",False)
# myTeamsPotentialAction3.addAction("HttpPost","Save","https://...")

# myTeamsMessage.addPotentialAction(myTeamsPotentialAction1)
# myTeamsMessage.addPotentialAction(myTeamsPotentialAction2)
# myTeamsMessage.addPotentialAction(myTeamsPotentialAction3)

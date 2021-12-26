global percentChanceOfExecution := 50

F2::
{
    if (shouldExecuteBasedOnChance()) {
        Msgbox, % Format("I executed with {}% of execution", percentChanceOfExecution)
    }
}

shouldExecuteBasedOnChance() {
    shouldExecute := false
    Random, randomNumber, 1, 100

    if (randomNumber <= percentChanceOfExecution) {
        shouldExecute := true
    }

    return shouldExecute
}
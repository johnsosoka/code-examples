global executeHour = 8

global OneMinuteMilliseconds := 60000
F1::
    loop {

        if (shouldExecuteBasedOnTime())
        {
            Msgbox, "Executing Script at Execution Hour"
        }
        Sleep OneMinuteMilliseconds*60
    }


shouldExecuteBasedOnTime() {
    shouldExecute := false
    if (A_Hour = executeHour) {
        shouldExecute := true
    }

    return shouldExecute
}

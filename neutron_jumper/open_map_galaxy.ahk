#Requires AutoHotkey v2.0
#SingleInstance Force

logPath := A_ScriptDir "\neutron_log.txt"
SendMode("Event")  ; More compatible with Elite Dangerous
F11::{
    SetKeyDelay(30, 30)
    Log("Hotkey triggered (F11)")

    Sleep(500)
    Send("1")
    Log("Sent 1")

    Sleep(500)
    Send("a")
    Log("Sent a")

    Sleep(500)
    Send("{Space}")
    Log("Sent Space")

    Sleep(3000)
    Send("{Left}")
    Log("Sent Left Arrow")

    Sleep(500)
    Send("{Right}")
    Log("Sent Right Arrow")

    Sleep(1500)
    Send("{Space}")
    Log("Sent Space again")

    Sleep(500)
    Send("^v")  ; Ctrl+V to paste
    Log("Pasted (Ctrl+V)")

    Sleep(500)
    Send("{Down}")
    Log("Sent Down Arrow")

    Sleep(1500)
    Send("{Space}")
    Log("Sent Space")

    #wiggle the mouse in the middle of the page and press and hold down the main mouse button for 1.5s then after 0.5s hit esc wait 0.2s then esc

    Sleep(3500)
    Send("{Right}")
    Sleep(500)
    Send("{Right}")
    Log("Sent Right Arrow twice")

    Loop 6 {
        Sleep(200)
        Send("{Down}")
    }
    Log("Sent Down Arrow x6")

    Sleep(500)
    Send("{Space}")
    Log("Final Space - Sequence complete")
}

Log(msg) {
    timestamp := FormatTime(, "yyyy-MM-dd HH:mm:ss")
    FileAppend("[" timestamp "] " msg "`n", "ahk_log.txt")
}

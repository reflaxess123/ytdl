# dl-youtube tab completion for PowerShell 7
# Добавь в свой $PROFILE:  . D:\CProjs\dl-youtube\dl-completion.ps1

function dl {
    uv run --project D:\CProjs\dl-youtube python D:\CProjs\dl-youtube\download.py @args
}

Register-ArgumentCompleter -CommandName dl -ScriptBlock {
    param($wordToComplete, $commandAst, $cursorPosition)

    $flags = @(
        @{Name='--mp3';       Tip='Скачать аудио MP3'}
        @{Name='--subs';      Tip='Скачать субтитры .srt'}
        @{Name='--text';      Tip='Субтитры -> чистый текст'}
        @{Name='--desc';      Tip='Описания видео -> текст'}
        @{Name='--list';      Tip='Список видео в CSV'}
        @{Name='--summary';   Tip='LLM-анализ лекции'}
        @{Name='--gemini';    Tip='Gemini 3.1 Pro вместо OpenRouter'}
        @{Name='--ru';        Tip='Прокси из YTDL_PROXY (для РФ)'}
        @{Name='-q';          Tip='Качество: 720 или 1080'}
        @{Name='--quality';   Tip='Качество: 720 или 1080'}
        @{Name='-o';          Tip='Папка для сохранения'}
        @{Name='--output';    Tip='Папка для сохранения'}
        @{Name='-p';          Tip='Прокси http://ip:port'}
        @{Name='--proxy';     Tip='Прокси http://ip:port'}
        @{Name='-c';          Tip='Куки из браузера'}
        @{Name='--cookies';   Tip='Куки из браузера'}
        @{Name='--subs-lang'; Tip='Язык субтитров (по умолч. ru)'}
        @{Name='--chunk';     Tip='Блок текста в минутах (по умолч. 5)'}
    )

    $tokens = $commandAst.ToString() -split '\s+'
    $usedFlags = $tokens | Where-Object { $_ -like '-*' }

    # Complete --quality values
    if ($tokens[-1] -in @('-q','--quality')) {
        '720','1080' | ForEach-Object {
            [System.Management.Automation.CompletionResult]::new($_, $_, 'ParameterValue', $_)
        }
        return
    }

    # Complete --cookies values
    if ($tokens[-1] -in @('-c','--cookies')) {
        'chrome','firefox','edge','brave','opera' | ForEach-Object {
            [System.Management.Automation.CompletionResult]::new($_, $_, 'ParameterValue', $_)
        }
        return
    }

    # Complete flags
    $flags | Where-Object {
        $_.Name -like "$wordToComplete*" -and $_.Name -notin $usedFlags
    } | ForEach-Object {
        [System.Management.Automation.CompletionResult]::new($_.Name, $_.Name, 'ParameterName', $_.Tip)
    }
}

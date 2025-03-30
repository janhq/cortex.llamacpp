@echo off

set "TEMP=C:\Users\%UserName%\AppData\Local\Temp"
set "MODEL_LLM_PATH=%TEMP%\testllm"
set "MODEL_EMBEDDING_PATH=%TEMP%\test-embedding"

rem Check for required arguments
if "%~3"=="" (
    echo Usage: %~0 ^<path_to_binary^> ^<url_to_download_llm^> ^<url_to_download_embedding^>
    exit /b 1
)

set "BINARY_PATH=%~1"
set "DOWNLOAD_LLM_URL=%~2"
set "DOWNLOAD_EMBEDDING_URL=%~3"

for %%i in ("%BINARY_PATH%") do set "BINARY_NAME=%%~nxi"

echo BINARY_NAME=%BINARY_NAME%

del %TEMP%\response1.log 2>nul
del %TEMP%\response2.log 2>nul
del %TEMP%\response4.log 2>nul
del %TEMP%\response5.log 2>nul
del %TEMP%\response6.log 2>nul
del %TEMP%\server.log 2>nul

set /a min=9999
set /a max=11000
set /a range=max-min+1
set /a PORT=%min% + %RANDOM% %% %range%

rem Start the binary file
start /B "" "%BINARY_PATH%" "127.0.0.1" %PORT% > %TEMP%\server.log 2>&1

ping -n 6 127.0.0.1 %PORT% > nul

rem Capture the PID of the started process with "server" in its name
for /f "tokens=2" %%a in ('tasklist /fi "imagename eq %BINARY_NAME%" /fo list ^| findstr /B "PID:"') do (
    set "pid=%%a"
)

echo pid=%pid%

if not defined pid (
    echo server failed to start. Logs:
    type %TEMP%\server.log
    exit /b 1
)

rem Wait for a few seconds to let the server start

rem Check if %TEMP%\testmodel exists, if not, download it
if not exist "%MODEL_LLM_PATH%" (
    curl.exe --connect-timeout 300 %DOWNLOAD_LLM_URL% --output "%MODEL_LLM_PATH%"
)

if not exist "%MODEL_EMBEDDING_PATH%" (
    curl.exe --connect-timeout 300 %DOWNLOAD_EMBEDDING_URL% --output "%MODEL_EMBEDDING_PATH%"
)

rem Define JSON strings for curl data
call set "MODEL_LLM_PATH_STRING=%%MODEL_LLM_PATH:\=\\%%"
call set "MODEL_EMBEDDING_PATH_STRING=%%MODEL_EMBEDDING_PATH:\=\\%%"
set "curl_data1={\"llama_model_path\":\"%MODEL_LLM_PATH_STRING%\"}"
set "curl_data2={\"messages\":[{\"content\":\"Hello there\",\"role\":\"assistant\"},{\"content\":\"Write a long and sad story for me\",\"role\":\"user\"}],\"stream\":false,\"model\":\"testllm\",\"max_tokens\":50,\"stop\":[\"hello\"],\"frequency_penalty\":0,\"presence_penalty\":0,\"temperature\":0.1}"
set "curl_data4={\"llama_model_path\":\"%MODEL_EMBEDDING_PATH_STRING%\", \"embedding\": true, \"model_type\": \"embedding\"}"
set "curl_data5={}"
set "curl_data6={\"input\": \"Hello\", \"model\": \"test-embedding\", \"encoding_format\": \"float\"}"
@REM set "curl_data7={\"model\": \"test-embedding\"}"

rem Print the values of curl_data for debugging
echo curl_data1=%curl_data1%
echo curl_data2=%curl_data2%
echo curl_data4=%curl_data4%
echo curl_data5=%curl_data5%
echo curl_data6=%curl_data6%
@REM echo curl_data7=%curl_data7%

rem Run the curl commands and capture the status code
curl.exe --connect-timeout 60 -o "%TEMP%\response1.log" -s -w "%%{http_code}" --location "http://127.0.0.1:%PORT%/loadmodel" --header "Content-Type: application/json" --data "%curl_data1%" > %TEMP%\response1.log 2>&1

curl.exe --connect-timeout 60 -o "%TEMP%\response2.log" -s -w "%%{http_code}" --location "http://127.0.0.1:%PORT%/v1/chat/completions" ^
--header "Content-Type: application/json" ^
--data "%curl_data2%" > %TEMP%\response2.log 2>&1

curl.exe --connect-timeout 60 -o "%TEMP%\response4.log" --request POST -s -w "%%{http_code}" --location "http://127.0.0.1:%PORT%/loadmodel" --header "Content-Type: application/json" --data "%curl_data4%" > %TEMP%\response4.log 2>&1

curl.exe --connect-timeout 60 -o "%TEMP%\response5.log" --request GET -s -w "%%{http_code}" --location "http://127.0.0.1:%PORT%/models" --header "Content-Type: application/json" --data "%curl_data5%" > %TEMP%\response5.log 2>&1

curl.exe --connect-timeout 60 -o "%TEMP%\response6.log" -s -w "%%{http_code}" --location "http://127.0.0.1:%PORT%/v1/embeddings" ^
--header "Content-Type: application/json" ^
--data "%curl_data6%" > %TEMP%\response6.log 2>&1

set "error_occurred=0"

rem Read the status codes from the log files
for /f %%a in (%TEMP%\response1.log) do set "response1=%%a"
for /f %%a in (%TEMP%\response2.log) do set "response2=%%a"
for /f %%a in (%TEMP%\response4.log) do set "response4=%%a"
for /f %%a in (%TEMP%\response5.log) do set "response5=%%a"
for /f %%a in (%TEMP%\response6.log) do set "response6=%%a"

if "%response1%" neq "200" (
    echo The first curl command failed with status code: %response1%
    type %TEMP%\response1.log
    set "error_occurred=1"
)

if "%response2%" neq "200" (
    echo The second curl command failed with status code: %response2%
    type %TEMP%\response2.log
    set "error_occurred=1"
)

if "%response4%" neq "200" (
    echo The fourth curl command failed with status code: %response4%
    type %TEMP%\response4.log
    set "error_occurred=1"
)

if "%response5%" neq "200" (
    echo The fifth curl command failed with status code: %response5%
    type %TEMP%\response5.log
    set "error_occurred=1"
)

if "%response6%" neq "200" (
    echo The sixth curl command failed with status code: %response6%
    type %TEMP%\response6.log
    set "error_occurred=1"
)

if "%error_occurred%"=="1" (
    echo Server test run failed!!!!!!!!!!!!!!!!!!!!!!
    echo Server Error Logs:
    type %TEMP%\server.log
    taskkill /f /pid %pid%
    exit /b 1
)


echo ----------------------
echo Log load llm model:
type %TEMP%\response1.log

echo ----------------------
echo Log run test:
type %TEMP%\response2.log

echo ----------------------
echo Log load embedding model:
type %TEMP%\response4.log

echo ----------------------
echo Log running models:
type %TEMP%\response6.log


echo ----------------------
echo Log run embedding test:
type %TEMP%\response6.log

echo Server test run successfully!

rem Kill the server process
@REM taskkill /f /pid %pid%
taskkill /f /im server.exe 2>nul || exit /B 0
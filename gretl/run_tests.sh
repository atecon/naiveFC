#!/bin/bash
set -e

DIR=$(dirname $(realpath "$0")) 	# locate folder where this sh-script is located in

PROJECT="naiveFC"

SCRIPT1="./tests/run_tests.inp"
SCRIPT2="./tests/run_tests_static_case.inp"
SCRIPT3="./tests/run_tests_static_case_with_exogenous.inp"
SCRIPT4="./tests/run_tests_movwindow_case.inp"

cd $DIR
echo "Switched to ${DIR}"


gretlcli -b -e -q ${SCRIPT1}
if [ $? -eq 0 ]
then
  echo "Success: Tests for script ${SCRIPT1} passed."
else
  echo "Failure: Tests for script ${SCRIPT1} not passed."
  exit 1
fi

gretlcli -b -e -q ${SCRIPT2}
if [ $? -eq 0 ]
then
  echo "Success: Tests for script ${SCRIPT2} passed."
else
  echo "Failure: Tests for script ${SCRIPT2} not passed."
  exit 1
fi


gretlcli -b -e -q ${SCRIPT3}
if [ $? -eq 0 ]
then
  echo "Success: Tests for script ${SCRIPT3} passed."
else
  echo "Failure: Tests for script ${SCRIPT3} not passed."
  exit 1
fi


gretlcli -b -e -q ${SCRIPT4}
if [ $? -eq 0 ]
then
   echo "Success: Tests for script ${SCRIPT4} passed."
else
  echo "Failure: Tests for script ${SCRIPT4} not passed." >&2
  exit 1
fi


exit 0

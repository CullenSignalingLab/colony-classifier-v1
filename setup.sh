#!/usr/bin/env bash
set -euxo pipefail

DB="ccc.db"
CSV="lab_classifications.csv"
TABLE="lab_classification"

if [ -f "$DB" ]; then
    rm "$DB"
fi

sqlite3 "$DB" "VACUUM;"

if [ -f "$CSV" ]; then
    # Get CSV header and build CREATE TABLE statement
    HEADER=$(head -n 1 "$CSV" | tr -d '\r')
    COLS=$(echo "$HEADER" | awk -F',' '{for(i=1;i<=NF;i++) printf "%s TEXT%s", $i, (i<NF?",":"")}')
    # Use HEADER for column names in both CREATE and INSERT
    CREATE_SQL="CREATE TABLE IF NOT EXISTS $TABLE (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        created TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        $COLS
    );"
    sqlite3 "$DB" "$CREATE_SQL"

    # Create temporary table for import
    TMP_TABLE="tmp_import"
    CREATE_TMP_SQL="CREATE TEMP TABLE $TMP_TABLE ($COLS);"
    sqlite3 "$DB" "$CREATE_TMP_SQL"

    # Prepare temp CSV without header
    tail -n +2 "$CSV" > tmp_lab_classifications.csv

    # Import into temporary table
    sqlite3 "$DB" <<EOF
.mode csv
.import tmp_lab_classifications.csv $TMP_TABLE
EOF

    # Insert from temporary table into main table (auto id/created)
    sqlite3 "$DB" "INSERT INTO $TABLE ($HEADER) SELECT * FROM $TMP_TABLE;"

    rm tmp_lab_classifications.csv
fi
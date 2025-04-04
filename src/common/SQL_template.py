query_summarization_template = """
WITH UPDATED_KJOINFO_TSUBAN AS (
    SELECT KJOINFO_TSUBAN
    FROM kujo.TBLKJ01 t1
    WHERE t1.FINALUPD_TSTNP >= '@last_run_date'::timestamp
        AND t1.FINALUPD_TSTNP < '@run_date'::timestamp
        AND t1.SAKUJO_FLG = '0'
        AND t1.kj_shrschkjky <> '本社最終確認済み'

    UNION

    SELECT KJOINFO_TSUBAN
    FROM kujo.TBLKJSESS t2
    WHERE t2.UPD_TIME_STANP >= '@last_run_date'::timestamp
        AND t2.UPD_TIME_STANP < '@run_date'::timestamp

    UNION

    SELECT KJOINFO_TSUBAN
    FROM kujo.TBLKJTOUJ t3
    WHERE t3.UPD_TIME_STANP >= '@last_run_date'::timestamp
        AND t3.UPD_TIME_STANP < '@run_date'::timestamp
)
, MOSD AS (
    SELECT
        KJOINFO_TSUBAN,
        MOSD_NAIYOU
    FROM
        kujo.TBLKJ01 t1
    WHERE
        KJOINFO_TSUBAN IN (SELECT KJOINFO_TSUBAN FROM UPDATED_KJOINFO_TSUBAN)
)
, KJSESS AS (
    SELECT
        KJOINFO_TSUBAN,
        STRING_AGG(TAIOUSHOUSAI, ' ' ORDER BY UPD_TIME_STANP ASC)
    AS TAIOUSHOUSAI
    FROM
        kujo.TBLKJSESS
    WHERE
        KJOINFO_TSUBAN IN (SELECT KJOINFO_TSUBAN FROM UPDATED_KJOINFO_TSUBAN)
    GROUP BY
        KJOINFO_TSUBAN
)
, KJTOUJ AS (
    SELECT
        KJOINFO_TSUBAN,
        STRING_AGG(KUJO_PARTYBIKO, ' ' ORDER BY UPD_TIME_STANP ASC)
    AS KUJO_PARTYBIKO
    FROM
        kujo.TBLKJTOUJ
    WHERE
        KJOINFO_TSUBAN IN (SELECT KJOINFO_TSUBAN FROM UPDATED_KJOINFO_TSUBAN)
    GROUP BY
        KJOINFO_TSUBAN
)
SELECT
    t1.KJOINFO_TSUBAN,
    t1.MOSD_NAIYOU,
    t2.TAIOUSHOUSAI,
    t3.KUJO_PARTYBIKO
FROM
    MOSD t1
LEFT JOIN
    KJSESS t2 ON t1.KJOINFO_TSUBAN = t2.KJOINFO_TSUBAN
LEFT JOIN
    KJTOUJ t3 ON t1.KJOINFO_TSUBAN = t3.KJOINFO_TSUBAN;
"""

query_classification_template = """
WITH MOSD AS (
    SELECT
        KJOINFO_TSUBAN
        , MOSD_NAIYOU
    FROM
        kujo.TBLKJ01 t1
    WHERE
        t1.FINALUPD_TSTNP >= '@last_run_date'::timestamp
        AND t1.FINALUPD_TSTNP < '@run_date'::timestamp
        AND t1.FINALUPOPSHZKC = 'batch'
        AND t1.KJNYTRKENDY4MD IS NULL
        AND t1.SAKUJO_FLG = '0'
        AND t1.KJ_SHRSCHKJKY <> '本社最終確認済み'
        AND NOT EXISTS (
            SELECT 1
            FROM kujo.TBLGENBNRUI t4
            WHERE t4.KJOINFO_TSUBAN = t1.KJOINFO_TSUBAN
        )
        AND MOSD_NAIYOU IS NOT NULL
        AND MOSD_NAIYOU <> ''
)
SELECT
    KJOINFO_TSUBAN,
    MOSD_NAIYOU,
    NULL AS TAIOUSHOUSAI,
    NULL AS KUJO_PARTYBIKO
FROM
    MOSD;
"""

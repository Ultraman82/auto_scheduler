WITH end_face_process as (SELECT
                              order_number,
                              operation_sequence,
                              facility_id
                              FROM manufacturing_order_processes
                              WHERE facility_id = 'C0040'),

     ranked_previous as (SELECT
                             ef.order_number as order_number,
                             ef.operation_sequence,
                             ef.facility_id,
                             mop.operation_sequence as previous_sequence,
                             mop.facility_id as previous_facility,
                             RANK()
                             OVER (PARTITION BY ef.order_number ORDER BY mop.operation_sequence DESC) as previous_ranked
                             FROM end_face_process ef
                                      LEFT JOIN manufacturing_order_processes mop
                                                ON ef.order_number = mop.order_number AND
                                                   CAST(mop.operation_sequence as INT) <
                                                   CAST(ef.operation_sequence AS INT)),

     end_face_out_data as (SELECT
                               rp.order_number,
                               operation_sequence,
                               rp.facility_id,
                               previous_sequence,
                               previous_facility,
                               previous_ranked,
                               mol.facility_id as mol_facility,
                               time_out as end_face_out
                               FROM ranked_previous rp
                                        LEFT JOIN manufacturing_order_logs mol ON rp.order_number = mol.order_number AND
                                                                                  (rp.facility_id = mol.facility_id OR
                                                                                   rp.previous_facility =
                                                                                   mol.facility_id)
                               WHERE previous_ranked = 1
                                 AND EXTRACT(year from time_out) = '2022'
                                 AND mol.facility_id = 'C0040'
                               ORDER BY rp.order_number),
     previous_facility_out_data as (SELECT
                                        rp.order_number,
                                        operation_sequence,
                                        rp.facility_id,
                                        previous_sequence,
                                        previous_facility,
                                        previous_ranked,
                                        mol.facility_id as mol_facility,
                                        time_out as previous_facility_out
                                        FROM ranked_previous rp
                                                 LEFT JOIN manufacturing_order_logs mol
                                                           ON rp.order_number = mol.order_number AND
                                                              (rp.facility_id = mol.facility_id OR
                                                               rp.previous_facility = mol.facility_id)
                                        WHERE previous_ranked = 1
                                          AND EXTRACT(year from time_out) = '2022'
                                          AND mol.facility_id = previous_facility
                                        ORDER BY rp.order_number),
     lead_times as (SELECT
                        ef.order_number,
                        ef.operation_sequence,
                        ef.facility_id,
                        ef.previous_sequence,
                        ef.previous_facility,
                        previous_facility_out,
                        end_face_out,
                        ROUND((EXTRACT(EPOCH FROM end_face_out) - EXTRACT(EPOCH FROM previous_facility_out)) / 3600,
                              2) as lead_time_hours,
                        end_face_out - previous_facility_out as lead_time
                        FROM end_face_out_data ef
                                 LEFT JOIN previous_facility_out_data pf ON ef.order_number = pf.order_number
                        ORDER BY ef.order_number)
SELECT
    COUNT(lt.order_number) as order_count,
    product_family,
    product_model,
    SUM(product_length * mano.order_quantity) as total_length,
    product_length,
    CASE
      WHEN  SUM(lead_time_hours / 24) !=  CAST(0 as numeric) THEN ROUND(CAST(SUM(product_length * mano.order_quantity) / SUM(lead_time_hours / 24) as numeric),2)
    ELSE 0
    END as mm_per_day,
    ROUND(AVG(lead_time_hours / 24), 2) avg_days,
    ROUND(AVG(lead_time_hours), 2) avg_hours,
    ROUND(SUM(lead_time_hours), 2) as sum_hours
    FROM lead_times lt
             LEFT JOIN manufacturing_orders mano ON lt.order_number = mano.order_number
    GROUP BY
            product_family,
             product_model,
             product_length
    ORDER BY avg_hours DESC;


    haas_sql = """
    WITH Ranked AS (
        SELECT
            p.order_number,
            p.operation_sequence,
            p.facility_id,
            ROW_NUMBER() OVER (PARTITION BY p.order_number ORDER BY p.operation_sequence) AS operation_sequence_rank
        FROM
            manufacturing_order_processes p
    ),
    FirstK0020 AS (
        SELECT
            r.order_number,
            r.operation_sequence,
            r.facility_id
        FROM
            Ranked r
        WHERE
            r.operation_sequence_rank = 1
            AND r.facility_id = 'K0020'
    )
    SELECT
        CAST(o.product_length as INT) as length, o.product_family as family, product_model as model, item_description as des, CAST(order_quantity as INT) as qty, o.order_number as mo, reference_number as hk, order_scheduled_due as due, product_g1 as g1, product_pitch as pitch
    FROM
        manufacturing_orders o
    INNER JOIN
        FirstK0020 f ON o.order_number = f.order_number
        WHERE o.order_release_code = 5
            AND o.order_status = '10'
            AND o.item_description LIKE '%+%'
            AND o.product_family = 'HSR'
            AND o.product_model IN ('45', '35')
            AND o.printed_due <= CURRENT_DATE + 168
            AND o.reference_number LIKE 'HK%';
    """   
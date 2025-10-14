// Dash clientside helpers for MLIP table alignment
window.dash_clientside = Object.assign({}, window.dash_clientside, {
  mlip: {

    measure_col_centers: function(columns, ts, metricList, tableId, overlayId) {
      try {
        const tbl = document.getElementById(tableId);
        const overlay = document.getElementById(overlayId);
        if (!tbl || !overlay || !Array.isArray(columns) || !Array.isArray(metricList)) {
          return window.dash_clientside.no_update;
        }

        const container = tbl.closest('.dash-table-container') || tbl;
        let headers = container.querySelectorAll('table thead th');
        if (!headers || headers.length === 0) {
          headers = container.querySelectorAll('.dash-spreadsheet-container thead th');
        }
        if (!headers || headers.length === 0) {
          headers = container.querySelectorAll('[role="columnheader"]');
        }
        if (!headers || headers.length === 0) {
          headers = container.querySelectorAll('th');
        }
        if (!headers || headers.length === 0) {
          return window.dash_clientside.no_update;
        }

        // Find the grid container (the shaded box)
        const gridContainer = document.getElementById(tableId + '-threshold-grid');

        const colIds = columns.map(c => c.id);
        const overlayRect = overlay.getBoundingClientRect();
        const centers = {};

        // Calculate grid container dimensions for proper centering
        let gridHeight = 0;
        let overlayOffsetTop = 0;
        if (gridContainer) {
          const gridRect = gridContainer.getBoundingClientRect();
          gridHeight = gridRect.height;
          overlayOffsetTop = overlayRect.top - gridRect.top;
        }

        for (const metric of metricList) {
          const idx = colIds.indexOf(metric);
          if (idx < 0 || idx >= headers.length) continue;

          const headerRect = headers[idx].getBoundingClientRect();
          const headerCenter = headerRect.left + (headerRect.width / 2);
          const relativeCenter = headerCenter - overlayRect.left;
          const percentage = (relativeCenter / overlayRect.width) * 100;

          centers[metric] = percentage;
        }

        // Add grid dimensions to the return object
        centers.__gridHeight = gridHeight;
        centers.__overlayOffsetTop = overlayOffsetTop;

        return centers;
      } catch (e) {
        return window.dash_clientside.no_update;
      }
    },

    align_thresholds: function(columns, tabValue, ts, tableId) {
      try {
        const tbl = document.getElementById(tableId);
        const grid = document.getElementById(tableId + '-threshold-grid');
        if (!tbl || !grid || !Array.isArray(columns) || columns.length === 0) {
          return window.dash_clientside.no_update;
        }
        const container = tbl.closest('.dash-table-container') || tbl;

        let headers = container.querySelectorAll('table thead th');
        if (!headers || headers.length === 0) {
          headers = container.querySelectorAll('.dash-spreadsheet-container thead th');
        }
        if (!headers || headers.length === 0) {
          headers = container.querySelectorAll('[role="columnheader"]');
        }
        if (!headers || headers.length === 0) {
          return window.dash_clientside.no_update;
        }

        const widths = Array.from(headers).map(h => Math.max(0, Math.round(h.getBoundingClientRect().width)));
        const template = widths.map(w => w + 'px').join(' ');
        const firstRect = headers[0].getBoundingClientRect();
        const contRect = container.getBoundingClientRect();
        const leftOffset = Math.max(0, Math.round(firstRect.left - contRect.left));

        return {
          display: 'grid',
          gridTemplateColumns: template,
          marginLeft: leftOffset + 'px'
        };
      } catch (e) {
        console.error('align_thresholds error', e);
        return window.dash_clientside.no_update;
      }
    },

    freeze_table_widths: function(columns, tabValue, ts, tableId) {
      try {
        const tbl = document.getElementById(tableId);
        if (!tbl || !Array.isArray(columns) || columns.length === 0) {
          return [window.dash_clientside.no_update, window.dash_clientside.no_update];
        }
        const container = tbl.closest('.dash-table-container') || tbl;

        let headers = container.querySelectorAll('table thead th');
        if (!headers || headers.length === 0) {
          headers = container.querySelectorAll('.dash-spreadsheet-container thead th');
        }
        if (!headers || headers.length === 0) {
          headers = container.querySelectorAll('[role="columnheader"]');
        }
        if (!headers || headers.length === 0) {
          return [window.dash_clientside.no_update, window.dash_clientside.no_update];
        }

        const widths = Array.from(headers).map(h => Math.max(0, Math.round(h.getBoundingClientRect().width)));

        const styles = [];
        for (let i = 0; i < Math.min(columns.length, widths.length); i++) {
          const col = columns[i];
          const colId = (col && col.id !== undefined) ? col.id : (col && col.name !== undefined ? col.name : String(i));
          styles.push({
            if: { column_id: colId },
            width: widths[i] + 'px',
            minWidth: widths[i] + 'px',
            maxWidth: widths[i] + 'px'
          });
        }
        const fillWidth = false;
        return [styles, fillWidth];
      } catch (e) {
        console.error('freeze_table_widths error', e);
        return [window.dash_clientside.no_update, window.dash_clientside.no_update];
      }
    },

    apply_widths_to_control: function(resultsColumns, ts, controlColumns, resultsId, controlsId) {
      try {
        const tbl = document.getElementById(resultsId);
        const ctrl = document.getElementById(controlsId);
        if (!tbl || !ctrl || !Array.isArray(resultsColumns)) {
          return [window.dash_clientside.no_update, window.dash_clientside.no_update];
        }
        const container = tbl.closest('.dash-table-container') || tbl;
        let headers = container.querySelectorAll('table thead th');
        if (!headers || headers.length === 0) {
          headers = container.querySelectorAll('.dash-spreadsheet-container thead th');
        }
        if (!headers || headers.length === 0) {
          headers = container.querySelectorAll('[role="columnheader"]');
        }
        if (!headers || headers.length === 0) {
          return [window.dash_clientside.no_update, window.dash_clientside.no_update];
        }
        const widths = Array.from(headers).map(h => Math.max(0, Math.round(h.getBoundingClientRect().width)));
        // Build style_cell_conditional for controls table. Align by index.
        const styles = [];
        for (let i = 0; i < Math.min(controlColumns.length, widths.length); i++) {
          const col = controlColumns[i];
          const colId = (col && col.id !== undefined) ? col.id : (col && col.name !== undefined ? col.name : String(i));
          styles.push({
            if: { column_id: colId },
            width: widths[i] + 'px',
            minWidth: widths[i] + 'px',
            maxWidth: widths[i] + 'px'
          });
        }
        const total = widths.reduce((a,b)=>a+b, 0);
        const tableStyle = { minWidth: total + 'px', width: total + 'px', overflowX: 'auto' };

        // Also install a ResizeObserver to re-apply widths on container resize
        if (!ctrl.__mlipControlRO) {
          const ctrlContainer = ctrl.closest('.dash-table-container') || ctrl;
          const resultsContainer = container;
          const applyInline = () => {
            try {
              // Re-measure
              let hs = resultsContainer.querySelectorAll('table thead th');
              if (!hs || hs.length === 0) return;
              const ws = Array.from(hs).map(h => Math.max(0, Math.round(h.getBoundingClientRect().width)));
              const sum = ws.reduce((a,b)=>a+b,0);
              // Apply to controls header cells
              const ch = ctrlContainer.querySelectorAll('table thead th');
              for (let i = 0; i < Math.min(ch.length, ws.length); i++) {
                ch[i].style.width = ws[i] + 'px';
                ch[i].style.minWidth = ws[i] + 'px';
                ch[i].style.maxWidth = ws[i] + 'px';
              }
              const cTable = ctrlContainer.querySelector('table');
              if (cTable) {
                cTable.style.width = sum + 'px';
                cTable.style.minWidth = sum + 'px';
              }
            } catch (e) {
              // ignore
            }
          };
          const ro = new ResizeObserver(() => applyInline());
          ro.observe(resultsContainer);
          ctrl.__mlipControlRO = ro;
          // Apply once now too
          applyInline();
        }
        return [styles, tableStyle];
      } catch (e) {
        console.error('apply_widths_to_control error', e);
        return [window.dash_clientside.no_update, window.dash_clientside.no_update];
      }
    }
  }
});

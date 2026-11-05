(() => {
    'use strict';
    const byId = id => document.getElementById(id);
    const idPattern = /^[0-9]{4}[A-Z]{4}[0-9]{2}$/;
    const state = {left: null, right: null, direction: 'nemeses', page: 1, sort: 'events'};
    let overviewController, compareController;
    let comparisonData = null;

    function element(tag, text, className) {
        const node = document.createElement(tag);
        if (text != null) node.textContent = text;
        if (className) node.className = className;
        return node;
    }

    function ensureComparisonSort() {
        // A running Flask worker can still serve its previously cached
        // template after the JS changes. Restore the new controls when that
        // older markup is loaded, so comparisons remain usable.
        const comparison = byId('comparison');
        let select = byId('comparison-sort');
        let help = byId('comparison-sort-help');
        if (!select) {
            const controls = element('div', null, 'nemesis-sort');
            const label = element('label', 'Sort results');
            label.htmlFor = 'comparison-sort';
            select = element('select');
            select.id = 'comparison-sort';
            for (const [value, text] of [['events', 'Default event order'],
                ['advantage', 'Percentage difference (high to low)']]) {
                const option = element('option', text);
                option.value = value;
                select.append(option);
            }
            controls.append(label, select);
            comparison.insertBefore(controls, help || comparison.querySelector('.nemesis-table-wrap'));
        }
        if (!help) {
            help = element('p', null, 'form-help');
            help.id = 'comparison-sort-help';
            help.hidden = true;
            comparison.insertBefore(help, comparison.querySelector('.nemesis-table-wrap'));
        }
        select.setAttribute('aria-describedby', help.id);
        return {select, help};
    }

    const comparisonSort = ensureComparisonSort();

    function showError(id, message = '') {
        byId(id).textContent = message;
        byId(id).hidden = !message;
    }

    async function getJSON(path, signal) {
        const res = await fetch(path, {signal});
        const data = await res.json().catch(() => ({}));
        if (!res.ok) throw new Error(data.error || 'Could not load comparisons. Please try again.');
        return data;
    }

    function updateURL() {
        const params = new URLSearchParams();
        if (state.left) params.set('id', state.left.wca_id);
        if (state.right) params.set('opponent', state.right.wca_id);
        if (state.direction !== 'nemeses') params.set('direction', state.direction);
        if (state.page > 1) params.set('page', state.page);
        if (state.sort !== 'events') params.set('sort', state.sort);
        history.replaceState(null, '', '/nemesis' + (params.size ? '?' + params : ''));
    }

    function chosen(inputId, person) {
        byId(inputId).value = person.name ? `${person.name} (${person.wca_id})` : person.wca_id;
    }

    function setupSearch(inputId, suggestionsId, formId, selectPerson, errorId) {
        const input = byId(inputId), suggestions = byId(suggestionsId);
        let timer, controller, selected = null;
        const close = () => { suggestions.hidden = true; };
        const select = person => {
            clearTimeout(timer);
            controller?.abort();
            selected = person;
            chosen(inputId, person);
            close();
            selectPerson(person);
        };
        input.addEventListener('input', () => {
            selected = null;
            clearTimeout(timer);
            controller?.abort();
            close();
            const query = input.value.trim();
            if (query.length < 2) return;
            timer = setTimeout(async () => {
                controller = new AbortController();
                try {
                    const data = await getJSON('/api/nemesis/search?' + new URLSearchParams({q: query}), controller.signal);
                    suggestions.replaceChildren();
                    for (const person of data.competitors) {
                        const button = element('button', null, 'nemesis-suggestion');
                        button.type = 'button';
                        button.append(element('strong', person.name),
                            element('span', `${person.wca_id} · ${person.country_id}`));
                        button.addEventListener('click', () => select(person));
                        suggestions.append(button);
                    }
                    if (!data.competitors.length) suggestions.append(element('p', 'No competitors found. Try a WCA ID.'));
                    suggestions.hidden = false;
                } catch (err) {
                    if (err.name !== 'AbortError') {
                        suggestions.replaceChildren(element('p', err.message));
                        suggestions.hidden = false;
                    }
                }
            }, 300);
        });
        input.addEventListener('keydown', event => {
            if (event.key === 'ArrowDown' && !suggestions.hidden) {
                event.preventDefault();
                suggestions.querySelector('button')?.focus();
            }
        });
        suggestions.addEventListener('keydown', event => {
            const buttons = [...suggestions.querySelectorAll('button')];
            const index = buttons.indexOf(document.activeElement);
            if (event.key === 'ArrowDown' || event.key === 'ArrowUp') {
                event.preventDefault();
                const next = event.key === 'ArrowDown' ? index + 1 : index - 1;
                if (next < 0) input.focus();
                else buttons[Math.min(next, buttons.length - 1)]?.focus();
            }
        });
        input.closest('form').addEventListener('keydown', event => {
            if (event.key === 'Escape') { close(); input.focus(); }
        });
        document.addEventListener('click', event => {
            if (!input.closest('form').contains(event.target)) close();
        });
        byId(formId).addEventListener('submit', event => {
            event.preventDefault();
            const value = input.value.trim().toUpperCase();
            const id = value.match(/\(([0-9]{4}[A-Z]{4}[0-9]{2})\)$/)?.[1] || value;
            if (idPattern.test(id)) select({wca_id: id});
            else if (selected && input.value.includes(selected.wca_id)) select(selected);
            else showError(errorId, 'Choose a competitor from the search results, or enter a full WCA ID.');
        });
    }

    async function loadOverview() {
        if (!state.left) return;
        overviewController?.abort();
        const controller = overviewController = new AbortController();
        showError('nemesis-error');
        byId('nemesis-status').textContent = 'Checking official personal bests…';
        byId('nemesis-overview').hidden = true;
        updateURL();
        try {
            const params = new URLSearchParams({direction: state.direction, page: state.page});
            const data = await getJSON(`/api/nemesis/${encodeURIComponent(state.left.wca_id)}?${params}`, controller.signal);
            state.left = data.competitor;
            chosen('competitor-search', state.left);
            byId('competitor-heading').textContent = state.left.name;
            byId('competitor-meta').textContent = `${state.left.wca_id} · ${state.left.country_id}`;
            byId('competitor-wca').href = 'https://www.worldcubeassociation.org/persons/' + encodeURIComponent(state.left.wca_id);
            byId('nemeses-count').textContent = data.counts.nemeses.toLocaleString();
            byId('nemesized-count').textContent = data.counts.nemesized.toLocaleString();
            const incoming = state.direction === 'nemeses';
            byId('show-nemeses').setAttribute('aria-pressed', String(incoming));
            byId('show-nemesized').setAttribute('aria-pressed', String(!incoming));
            byId('nemesis-list-heading').textContent = incoming ? 'Their nemeses' : 'Competitors they nemesize';
            byId('nemesis-list').replaceChildren();
            for (const person of data.competitors) {
                const row = element('li');
                const description = element('div');
                const profile = element('a', person.name);
                profile.href = '/nemesis?' + new URLSearchParams({id: person.wca_id});
                description.append(profile, element('span', `${person.wca_id} · ${person.country_id}`));
                const button = element('button', 'Compare', 'nemesis-button secondary');
                button.type = 'button';
                button.setAttribute('aria-label', `Compare with ${person.name}`);
                button.addEventListener('click', () => {
                    state.right = person;
                    chosen('opponent-search', person);
                    loadComparison();
                    byId('h2h-heading').scrollIntoView({block: 'start'});
                });
                row.append(description, button);
                byId('nemesis-list').append(row);
            }
            const total = data.counts[state.direction];
            byId('nemesis-empty').hidden = data.competitors.length > 0;
            byId('nemesis-empty').textContent = !data.has_results
                ? 'This competitor has no valid ranked singles or averages.'
                : total === 0
                    ? (incoming ? 'No nemeses found.' : 'No competitors nemesized.')
                    : 'No results on this page. Use Previous to go back.';
            const pages = Math.max(1, Math.ceil(total / data.page_size));
            byId('nemesis-pagination').hidden = pages <= 1 && state.page === 1;
            byId('nemesis-page-label').textContent = `Page ${state.page.toLocaleString()} · ${total.toLocaleString()} competitors`;
            byId('nemesis-prev').disabled = state.page <= 1;
            byId('nemesis-next').disabled = state.page >= pages;
            byId('nemesis-overview').hidden = false;
            byId('nemesis-status').textContent = `${state.left.name}: ${data.counts.nemeses.toLocaleString()} nemeses, ${data.counts.nemesized.toLocaleString()} competitors nemesized.`;
        } catch (err) {
            if (err.name !== 'AbortError') {
                showError('nemesis-error', err.message);
                byId('nemesis-status').textContent = '';
            }
        }
    }

    function resultMagnitude(value, eventId) {
        if (typeof value === 'number') return Number.isFinite(value) && value > 0 ? value : null;
        if (typeof value !== 'string') return null;
        const text = value.trim().replace(eventId === '333fm' ? /\s+moves$/ : /s$/, '');
        const pattern = eventId === '333fm' ? /^\d+(?:\.\d+)?$/
            : /^\d+(?::\d{1,2}){0,2}(?:\.\d+)?$/;
        if (!pattern.test(text)) return null;
        const parts = text.split(':').map(Number);
        if (parts.slice(1).some(part => part >= 60)) return null;
        const magnitude = parts.reduce((total, part) => total * 60 + part, 0);
        return Number.isFinite(magnitude) && magnitude > 0 ? magnitude : null;
    }

    function percentageDifference(result, eventId) {
        if (eventId === '333mbf') return null;
        if (Number.isFinite(result.percentage_difference)) return result.percentage_difference;
        // Older running workers return formatted PBs without the percentage
        // field. Derive the same ratio from those PBs instead of leaving every
        // row unranked. Seconds, minute times, and FMC moves need distinct units.
        const left = resultMagnitude(result.left, eventId);
        const right = resultMagnitude(result.right, eventId);
        return left == null || right == null ? null : 100 * (right - left) / right;
    }

    function renderComparisonRows() {
        if (!comparisonData) return;
        const sorted = state.sort === 'advantage';
        const data = comparisonData;
        const rows = data.events.flatMap(event => {
            const kinds = event.event_id === '333mbf' ? ['single'] : ['single', 'average'];
            return kinds.map(kind => {
                const result = event[kind];
                return {event, kind, result, percentage: percentageDifference(result, event.event_id)};
            });
        });
        if (sorted) {
            rows.sort((a, b) => {
                const first = a.percentage;
                const second = b.percentage;
                if (first == null) return second == null ? 0 : 1;
                if (second == null) return -1;
                return second - first;
            });
        }
        comparisonSort.help.hidden = !sorted;
        comparisonSort.help.textContent = sorted
            ? `From ${data.left.name}’s perspective: wins first, then ties, then losses. Percentages are relative to ${data.right.name}’s result. Missing results and Multi-Blind have no percentage and appear last.`
            : '';
        byId('comparison-table-body').replaceChildren();
        for (const {event, kind, result, percentage} of rows) {
            const eventName = event.event_name.replace(/(\d)x\1x\1/g, '$1x$1').replace(/ Cube$/, '');
            const row = element('tr'), label = element('th', `${eventName} ${kind}`);
            label.scope = 'row';
            if (sorted) {
                let difference = 'Percentage unavailable';
                if (percentage != null) {
                    const magnitude = Math.abs(percentage);
                    const amount = magnitude.toLocaleString(undefined, {maximumSignificantDigits: 3});
                    difference = percentage === 0 ? '0% · tied'
                        : `${percentage > 0 ? '+' : '−'}${amount}% · ${percentage > 0 ? 'ahead' : 'behind'}`;
                }
                label.append(element('span', difference, 'nemesis-percentage'));
            }
            row.append(label);
            for (const side of ['left', 'right']) {
                const wins = result.winner === side;
                const cell = element('td', result[side] || '—', wins ? 'nemesis-winning-result' : '');
                if (wins) cell.append(element('span', 'Better', 'nemesis-win-label'));
                row.append(cell);
            }
            byId('comparison-table-body').append(row);
        }
    }

    async function loadComparison() {
        compareController?.abort();
        const controller = compareController = new AbortController();
        comparisonData = null;
        byId('comparison').hidden = true;
        showError('compare-error');
        if (!state.left || !state.right) {
            byId('compare-status').textContent = '';
            if (state.right) showError('compare-error', 'Choose the first competitor above to start a comparison.');
            return;
        }
        byId('compare-status').textContent = 'Comparing personal bests…';
        updateURL();
        try {
            const data = await getJSON('/api/nemesis/compare?' + new URLSearchParams({
                left: state.left.wca_id, right: state.right.wca_id,
            }), controller.signal);
            state.right = data.right;
            chosen('opponent-search', state.right);
            byId('comparison-heading').textContent = `${data.left.name} vs ${data.right.name}`;
            byId('comparison-caption').textContent = `${data.left.wca_id} vs ${data.right.wca_id} · Singles & averages`;
            byId('comparison-relationship').textContent = data.left_nemesizes_right
                ? `${data.left.name} nemesizes ${data.right.name}.`
                : data.right_nemesizes_left
                    ? `${data.right.name} nemesizes ${data.left.name}.`
                    : 'Neither competitor nemesizes the other.';
            const scores = data.scores;
            byId('comparison-score').textContent = `${data.left.name}: ${scores.left} wins · ${data.right.name}: ${scores.right} wins · ${scores.ties} ties · ${scores.unmatched} comparisons with a missing result`;
            const header = element('tr');
            const eventHeader = element('th', 'Event');
            eventHeader.scope = 'col';
            header.append(eventHeader);
            for (const person of [data.left, data.right]) {
                const th = element('th', person.name);
                th.scope = 'col';
                header.append(th);
            }
            byId('comparison-table-head').replaceChildren(header);
            comparisonData = data;
            renderComparisonRows();
            byId('comparison').hidden = false;
            byId('compare-status').textContent = '';
        } catch (err) {
            if (err.name !== 'AbortError') {
                showError('compare-error', err.message);
                byId('compare-status').textContent = '';
            }
        }
    }

    setupSearch('competitor-search', 'competitor-suggestions', 'nemesis-form', person => {
        state.left = person; state.page = 1;
        loadOverview(); loadComparison();
    }, 'nemesis-error');
    setupSearch('opponent-search', 'opponent-suggestions', 'opponent-form', person => {
        state.right = person; loadComparison();
    }, 'compare-error');
    comparisonSort.select.addEventListener('change', event => {
        state.sort = event.target.value;
        renderComparisonRows();
        updateURL();
    });
    for (const direction of ['nemeses', 'nemesized']) {
        byId('show-' + direction).addEventListener('click', () => {
            state.direction = direction; state.page = 1; loadOverview();
        });
    }
    byId('nemesis-prev').addEventListener('click', () => { state.page--; loadOverview(); });
    byId('nemesis-next').addEventListener('click', () => { state.page++; loadOverview(); });

    const initial = new URLSearchParams(location.search);
    if (initial.get('sort') === 'advantage') state.sort = 'advantage';
    comparisonSort.select.value = state.sort;
    if (initial.get('direction') === 'nemesized') state.direction = 'nemesized';
    const initialPage = Number(initial.get('page'));
    if (Number.isInteger(initialPage) && initialPage >= 1 && initialPage <= 100000) state.page = initialPage;
    for (const [param, side, input] of [['id', 'left', 'competitor-search'], ['opponent', 'right', 'opponent-search']]) {
        const id = (initial.get(param) || '').toUpperCase();
        if (idPattern.test(id)) { state[side] = {wca_id: id}; chosen(input, state[side]); }
    }
    if (state.left) { loadOverview(); loadComparison(); }
})();

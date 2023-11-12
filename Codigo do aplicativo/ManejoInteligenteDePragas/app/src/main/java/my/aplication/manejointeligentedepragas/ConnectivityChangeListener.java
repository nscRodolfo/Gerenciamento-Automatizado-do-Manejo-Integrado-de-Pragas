package my.aplication.manejointeligentedepragas;

import com.zplesac.connectionbuddy.models.ConnectivityEvent;

public interface ConnectivityChangeListener {

    /**
     * Interface method which is called when there is change in internet connection state.
     *
     * @param event ConnectivityEvent which holds all data about network connection state.
     */

    void onConnectionChange(ConnectivityEvent event);
}
